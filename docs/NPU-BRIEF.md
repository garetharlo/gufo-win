# NPU / perf brief

Handoff note so a fresh session can work on the next item without re-deriving
context. Everything here was verified against the tree at `a43391cd`; anything
that is not in section 1 was not confirmed.

## 1. Verified state

- Upstream source: `haloforge-labs/bubo`. Code tip `51d1495` is what we synced
  in `83dd5da3`; the later `1d744ad` (v0.1.1) touches docs only.
  `src/models/qwen38_flash_next/npu` (54 files) vs bubo `src/npu` (34): shared
  files are identical apart from namespace/include rewrites. Ours adds
  guard, chip power, fault telemetry, runtime status, platform info, chunk
  retry, mem trace, and rewrites `invalid_bits` as a bitset and
  `QualifiedChunk` extraction.
- NPU prefill is strictly opt-in: `GUFO_NPU_PREFILL=1` (`engine.cpp:242-270`).
  Upstream serves it by default. Knob defaults match upstream (`SLICES=6`,
  `GATE=0`, `SHEXP=1`, `TIMEOUT=8000`, keepalive 1 s).
- Executor overlap is preserved: `BeginLinearZ` (`executor.cpp:2095`) → GPU
  prefix `DenseF16SsmGemm` (2113) → `FinishLinearZ` (2164); the MoE
  shared-expert wait lands after routed prep.
- Guard (`guard_config.hpp`, `guard.cpp`, `staging.cpp`) is ours only. It is an
  *envelope* guard: 82 W whole-chip RAPL mean over 3 consecutive chunks (or 98 %
  memory commit) trips a 600 s NPU hold; held chunks simply run on the GPU. It
  never throttles the GPU. `GUFO_NPU_GUARD=on|off|log`, default `on`.
- Whole-chip RAPL cannot attribute power to the NPU. That is documented and
  benign on this host (see section 3); proving anything better needs a
  power-capped host.

## 2. False leads — do not re-investigate

Each of these was checked and is wrong:

- `GUFO_NPU_GUARD` is read and honoured; there is no `GUFO_POWER_LIMIT_W` and no
  `guard_active()`.
- There is no `NPU_PREFILL_DEFAULT` constant; NPU prefill is opt-in.
- There is no NPU busy/duty sampling anywhere in the npu tree.
- `tools/bench/npu_sweep.sh` does not exist and never existed.
- The 4.6 s idle gaps in the depth-32K trace are GPU prefill chunks. The PLE
  gather is hidden behind them: 131,072 rows per chunk, 344 distinct, 78 ms of
  reads on the first chunk and zero after. The trace's GPU-busy percentage
  counts decode-cycle work only, so it reads 0.2 % during a 32K prefill.

## 3. Measurements

pp8192, UD-Q4_K_XL, 32 GB BOSGAME host (`-d 0 -n 8 -r 1`, 2026-10-28, after
`1018b08f` moved prefill to 8192-token chunks):

| build                        | tok/s   |
| ---------------------------- | ------- |
| GPU only                     | 1667.40 |
| NPU + guard                  | 1566.91 |
| NPU, guard off               | 1532.60 |

The NPU path is 6-8 % slower than the GPU here, so it stays opt-in. The guard
trips as designed on this host (`chunk_means_w=[107.3, 106.0, 85.4] > 82.0`) and
the run then lands on the GPU, which is why guard-on beats guard-off by 2.2 %:
on an uncapped host the envelope guard is itself the fastest place to fail.

Older numbers (1608.09 / 1563.61 / 1527.68, 4096-token chunks) in
`docs/LINUX-PERF-PORT.md` are ~3.7 % stale but keep the same ordering.

Two runtime traps on this host:

- `build/release/gufo.exe` looks for artifacts in `<exe>/../tools/npu`, which the
  CMake tree does not have, so the NPU silently falls back to the GPU with
  `artifact open failed`. Pass `GUFO_NPU_DIR=<repo>/tools/npu` for dev builds.
  The default is correct for the installed layout (`result/bin/gufo`).
- `GUFO_NPU_PREFILL=1` alone is not enough to see NPU numbers; it also needs the
  XRT artifacts above.

Upstream's claimed +5-7 % at 8K and +3.5-5.5 % at 16K was measured on a 70 W
128 GB laptop. Here a 4096-token NPU slice costs about 2.6 s against a 2.55 s
GPU chunk, so there is no overlap win to collect on an uncapped host.

## 4. Work list

1. NPU recalibration on this host, after any prefill-chunk change, to refresh
   the guard numbers in section 3.

Closed as false leads: the PLE gather (see section 2) and the MTP verify-row
marginal. Width is not saturated with depth, and the default already runs at
the top of the curve:

| draft tokens | tg128 @ d0 | tg128 @ d32768 |
| ------------ | ---------- | -------------- |
| 1            | 41.5       | –              |
| 2            | 57.5       | 57.2           |
| 3            | 57.4       | 69.7           |
| 4            | 59.3       | 77.3           |
| 5            | 57.9       | 83.8           |
| 6            | 57.6       | –              |
| 7 (default)  | –          | 100.2          |

At depth the extra row pays for itself; at depth 0 it does not, so the flat part
of the curve is correct behaviour, not a scheduling bug. Logits are sampled on
device per row (`SupportsDeviceResidentSampling`), so there is no per-row D2H to
remove.

## 5. Checklist before claiming progress

- Re-read the code at the cited lines; line numbers drift.
- Measure with `result/bin/gufo` or `build/release/gufo`, one trace command,
  same model revision.
- Do not treat a skipped model as a pass.
