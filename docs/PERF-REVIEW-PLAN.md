# Performance Review Plan

Status: active plan, 2026-10-23.

How to find real performance issues in this codebase. This complements
`docs/PERFORMANCE.md` (measurement rules) — nothing here overrides it.

## Principles

1. Measure first. A change without a before/after number on a named workload
   is not a performance change, it is a guess.
2. Production target only: Linux x86-64 Strix Halo `gfx1151`. Windows-host
   numbers are directional, never promotional.
3. One mechanism per change, one commit, revert line in the message.
4. Static reading finds candidates; profiling convicts them. Most static
   "wins" in this repo have regressed when measured (see
   `docs/LINUX-PERF-PORT.md` history).

## Ordered workflow

### Step 1 — Pick one named workload

Choose from the bench suite (`gufo bench`, serving benchmark) and freeze it:
same model artifact, quant, context, prompt, seed, and token counts. Record
it in the change description. Candidates:

- pp2048 prefill (known gap: ~18-23% behind upstream — highest-value target)
- d32768 decode, greedy and sampled+MTP variants
- model load time (I/O bound; verify before optimizing)

### Step 2 — Profile the workload, not the code

- `tools/prof/prof.py` — pipeline/wall-time phases
- `tools/prof/isa_mix.py` — instruction mix for kernel suspects
- `tools/bench/gfx1151_peak.hip` — hardware ceiling to classify the limiter
  (bandwidth / compute / launch / sync / scheduling)

Take the top cost center only. Stop if it is <10% of wall time.

### Step 3 — Classify before touching

| Limiter | Typical fix location | Usually NOT worth it |
|---|---|---|
| Bandwidth | kernel data layout, copies, precision | instruction micro-tuning |
| Compute | kernel math, occupancy | CPU-side refactors |
| Launch | batching, graphs | single-kernel speed |
| Sync | executor polling (`executor.cpp`), event waits | more threads |
| Scheduling | runner pool, request packing (`src/cli/serve`) | kernel work |

CPU-side hot paths worth knowing: `src/core/sampling.cpp` (full-vocab
softmax per verify row), `speculative_verifier.cpp` verify loop,
tokenizers (decode path). GPU kernels live under
`src/models/*/kernels/rocm/` and are gated on the `optimize-kernel` skill.

### Step 4 — Focused reproducer + oracle

Standalone HIP experiment in `tools/bench/build.sh` harness or a CPU
microbench. Correctness oracle: existing parity/unit tests plus bit-exact
output comparison on a fixed seed. Reject on any mismatch.

### Step 5 — Gate and land

- Focused test for the touched path (e.g. `logit_sampler_test` for sampling)
- Full workload re-run vs baseline; keep only measured wins
- `tools/ci/check-format.py` before commit
- Conventional Commit, one line; revert instruction in the body
- End-to-end numbers recorded in `docs/models/<model>/BENCHMARKS.md`,
  not the experiment diary

## Anti-checklist (already ruled out by review, 2026-10)

- Static sweeps of HIP kernels without a profile — regressions historically
- `hipGetDeviceProperties`/malloc/event sites — all init-time
- GGUF reader I/O — mmap-based, load-time only
- Executor `hipStreamSynchronize` polls — deliberate graph-completion design
- Micro container swaps (hash map vs sorted vector) in sampling — measured
  noise-level; the cost is streaming the vectors, not the lookups

## Current state (2026-10-23)

- Committed: `e5c561f` `SampleResidual` skips sorted rebuild (less work per
  rejected draft; end-to-end delta unverified — needs Step 1/5 run).
- Open target: pp2048 prefill gap vs upstream.

## Static review: `src/core/hip` + `qwen38_flash_next` (2026-10-25)

Read every file in both trees without a profile, looking for work that is
provably redundant rather than merely unmeasured. Two changes were justified
as strictly-less-work with bit-identical output; both are CPU-side, so neither
needs a GPU parity run, only the existing unit/parity tests.

Landed:

- `ngram.cpp` `HashNgramRows` — the per-order XOR mix was re-summed from
  `ctx[0]` for every n-gram order, costing `n(n-1)/2` multiplies per token.
  The mix for order `k` is order `k-1`'s mix plus one term, so it now grows
  incrementally: `n-1` multiplies, identical terms in identical order. Runs
  once per token of every prefill chunk and every decode step.
- `engine.cpp` `DecodeBatchImpl` — the drafting round loop declared
  `heads`/`bodies` inside the loop body, so each of up to ~7 rounds per decode
  cycle heap-allocated two vectors. Both now live outside and are cleared per
  round.

Examined and deliberately left alone (with the reason, so this is not
re-litigated):

- `prompt_lookup.hpp` `Find` — the `at(i)` branch per backward comparison
  costs ~0.1 µs per proposal and splitting the tail/committed runs would add an
  invariant to a correctness-critical path.
- `executor.cpp` `Run` double lookup (`warmed_` + `graphs_`) — required, a key
  can be warmed eagerly and captured later.
- `executor.cpp` `WaitFlag` `steady_clock::now()` per poll — CPU burn, not
  latency; polling is the deliberate graph-completion design.
- `executor.cpp` `std::function` layer bodies and `CheckCancellation` — sub-µs
  against a ~25 ms decode cycle.
- `DecodeBatchImpl` `std::any_of` over ≤8 requests — 64 pointer compares.
- `cpu_ops.cpp` `Rope` recomputing `powf` per head — reference/oracle path.
- `kernels.hip.cpp` RoPE `powf` sites — ~0.5 µs/token; `mmq/` is vendor code.
- `mtp_policy.hpp`, `mtp_sampling.hpp`, `draft_vocab.hpp`,
  `platform/tuning.hpp`, `core/hip/detail/dispatch_telemetry.hpp` — already
  cached/precomputed or single-pass.

Conclusion: inside this scope the remaining static candidates are below
measurement noise. The real gap (pp2048 prefill) needs Step 2 profiling, not
more reading.
