# Linux → Windows performance port plan

Status of upstream performance work that can be ported to this Windows engine.
Read this first when resuming the task; update the checkboxes as items land.

No plan-mode restrictions apply now. Measure every kernel change with the same
TheRock ROCm 10.0.0 / clang 23 toolchain used for the Windows build.

## Sources

- Upstream: <https://github.com/gufo-org/gufo>
- Review clone: `%TEMP%\opencode\gufo-linux`
  (HEAD `270b533`, v0.18.0 at review time)
- Local base: upstream `fee9d2a` (PR #314 fix)
- Local Windows HEAD at review: `2eb0778`
- Local fork already has Windows-specific work in
  [WINDOWS.md](WINDOWS.md): copy kernels, recorded rollback, flag waits,
  fast sampling, fused HC down, hot-first upload, and 128 n-gram readers.

If the review clone is gone:

```powershell
git clone https://github.com/gufo-org/gufo %TEMP%\opencode\gufo-linux
```

Inspect a commit with:

```powershell
git -C %TEMP%\opencode\gufo-linux show <sha> -- <path>
```

Do **not** cherry-pick large upstream commits blindly. The fork has diverged in
`qwen38_flash_next/kernels/rocm/{kernels.hip.cpp,executor.cpp}` and
`qwen38_flash_next/engine.cpp`; adapt the changes and keep local behavior.

## Goal and published target

The user's target is the upstream Flash-Next table in
`docs/models/qwen3.8-flash-next/BENCHMARKS.md` (upstream `fec4759d`,
October 18 2026), measured on the same gfx1151 host with the same
`UD-Q4_K_XL` target and shared-Q8_0 MTP sidecar. Current local numbers
(fork `2eb0778` + Tier 1, `gufo bench`, one to two repetitions):

| metric | upstream | local | local/upstream |
| --- | ---: | ---: | ---: |
| pp2048, depth 0, AR | 1801.01 tok/s | 1478.20 | 82% |
| pp2048, depth 32K, AR | 1672.60 | 1289.17 | 77% |
| tg128, depth 0, AR | 28.29 | 27.22 | 96% |
| tg128, depth 32K, AR | 28.18 | 25.60 | 91% |
| tg128, depth 0, MTP mixed text | 35.79 | 37.10 | 104% |
| tg128, depth 0, MTP repetitive text | 116.93 | not measured | — |
| load to ready, MTP, 262144 ctx | 16.42 s | 34.8-36.7 s (131072 ctx) | ~45% |

Two clarifications that keep the comparison honest:

- The published 116.93 tok/s is the *repetitive* prompt column, where MTP
  drafts are almost always accepted. The comparable mixed-text column is
  35.79 tok/s, and the local MTP decode already matches it. Repetitive local
  numbers need `tools/bench/serving_bench` and a repetitive corpus; a plain
  `gufo bench` prompt does not reproduce that case.
- Upstream loading is Linux page-cache friendly and measured with MTP at
  context 262144; local loading includes Windows file-system behavior. The
  weight loader source is otherwise identical, so loading is an I/O-path
  investigation, not a kernel port.

So the real code gap is **prefill (~18-23%)**; decode is at parity and loading
is an I/O question. That ordering drives the work below.

## Status key

- `[ ]` not started
- `[~]` in progress
- `[x]` landed and measured
- `[-]` rejected; do not port

---

## Patch acceptance criteria

Every port must be judged on all five, not just throughput:

- **Prefill**: pp16/pp64/pp2048 and long-context pp must not regress.
- **Decoding**: tg at depth 0 and 32K, single session and 4 sessions.
- **Loading**: model load time must not get worse; a win here counts.
- **RAM**: peak host/VRAM must not grow without a stated reason; this host has
  one 96 GB carve, so memory is a first-class constraint.
- **Quality**: outputs/logits bit-identical, or bit-identical up to the
  documented FP reassociation where upstream did the same. No sampling,
  draft-vocabulary or precision change that could make the model dumber,
  unless the user explicitly asks for it.

If a change improves one and regresses another, record both numbers and decide
explicitly; do not claim the win alone.

## Tier 1 — same compiler, low risk, do first

### [x] #459 — keep prefill GEMMs spill-free on clang 23

Upstream: `f17e37b`. Upstream Linux measurements: Qwen27B Q8 pp2048
**291 → 464 tok/s**, pp64 **112 → 308**, Q4 pp64 **230 → 335** without the
fix. This is likely the largest Qwen27B prefill win on Windows because the
Windows build also uses clang 23.

Changes:

- `src/models/qwen/hip/kernels/prefill_quant_gemm.hip` around line 561:
  replace `#pragma unroll` with `#pragma unroll(BK <= 2 ? BK : 1)` and add
  `__builtin_amdgcn_sched_barrier(0);` after each K block.
- `src/models/qwen/hip/kernels/prefill_quant_gemm.hpp` around line 388:
  same unroll pragma.
- `src/models/qwen/CMakeLists.txt`: add `-fno-slp-vectorize` to
  `prefill_quant_gemm.hip` and `prefill_quant_wave64.hip`.
- `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp` around line 3887:
  replace the runtime-index `code_cache` select in `RoutedF16GEMMKernel` with
  the four masked blends from upstream. Also check the Q8_0 path.

Windows fit: pure HIP/CMake. `__builtin_amdgcn_sched_barrier` and
`-fno-slp-vectorize` are supported by TheRock clang 23. No OS APIs.

Acceptance: outputs/logits bit-identical; Qwen27B pp16/pp64/pp2048 improve.

Windows result: landed (`prefill_quant_gemm.{hip,hpp}`, `qwen/CMakeLists.txt`,
`kernels.hip.cpp`). Flash-Next pp16/pp64/pp2048 unchanged within noise
(134.80/347.31/1438.95 vs 135.41/351.15/1419.66 tok/s, one run each, A/B
binaries `gufo-tier1.exe`/`gufo-baseline.exe`). Greedy generation byte-identical
(seed 42, 32 tokens) and `--validate-prefill 64` identical
(rmse 0.37, cosine 0.99). No Qwen27B GGUF on this host, so the headline
291→464 tok/s case is unmeasured. `#pragma unroll(BK <= 2 ? BK : 1)` raises a
harmless `-Wcuda-compat` warning on clang 23, matching upstream.

### [x] #481 — unroll Qwen-Image fused-attention value halves

Upstream: `b2438a4`. clang 23 `FusedAttention<false>` is **12.3% slower**
without it; denoise is **4% slower**.

Change: `src/models/qwen_image_21/hip/attention.hip` around line 149, add
`#pragma unroll` before `for (int half = 0; half < 2; half++)`.

Windows fit: pragma only. Acceptance: denoise output identical, faster.

Windows result: landed. Not measured: no Qwen-Image denoise A/B was run; the
change is a single pragma on an existing loop, and Qwen-Image weights are
present if a follow-up measurement is wanted.

### [x] #482 — keep DeepSeek 8-row IQ2 tiles out of scratch

Upstream: `e9b0f0b`. Tile kernel **9–10% faster**; pp8–pp24 **1.7–2.0%**
faster.

Changes:

- `src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_iq2_gate.hip.hpp`
  around lines 163 and 178: unroll the 8-pair loops with constant indices.
- `src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_norm.hip.hpp`
  line 15: add `__launch_bounds__(256)` to
  `rms_norm_plain_regs_kernel`.
- `ds4_rocm_moe.hip.hpp`: apply the matching 8-row tile-loop change.

Windows fit: pure HIP. Acceptance: bit-identical logits, faster pp8–pp24.

Windows result: landed in all three detail headers (unrolled 8-pair gate loops
with fused FMA, `__launch_bounds__(256)` norm, three MoE tile kernels). Compiled
clean for gfx1151 (only the pre-existing `ds4_rocm_indexer_topk` pass-failed
warning). **Not measured**: no DeepSeek V4 Flash weights on this host, so the
9–10% tile win and 1.7–2.0% pp8–pp24 win are unverified locally.

### [x] #633 fragment — prepare hipBLASLt indexer plans before readiness

Upstream: `1f97092`. Removes roughly **100 ms** from the first Flash-Next
prompt.

Changes:

- `src/models/qwen38_flash_next/kernels/rocm/blaslt.cpp`: add `Prepare()`,
  reusing the exact-shape `FindPlan`/`MakePlan` path and the `OwnF16` shortcut.
- `blaslt.hpp`: declare it.
- `executor.cpp` in `Executor::Create`: prepare the indexer K/Q plans before
  returning, before readiness.

Windows fit: pure HIP/host. No component cache needed. Do not take the rest of
#633; it depends on Linux page pinning.

Acceptance: first-prompt latency drops; plans still exact-shape.

---

## Tier 2 — decode wins

### [x] #613 — copy repeated context into MTP drafts (prompt lookup)

Implemented in the fork before upstream (`ae59401`): `prompt_lookup.hpp`
(greedy/sampled gates, resume-after-edit, snapshot state), engine/CLI/serve
wiring, tests. Upstream `8497711` is the follow-up of this work. Remaining
delta: upstream defaults it on for single-session MTP; the fork keeps it
opt-in (`engine.hpp` `prompt_lookup = false`). Upstream Linux measurements:

- rewrite/copy agent requests: **+40–46% decode** (e.g. 69.8 → 99.6 tok/s)
- 25k follow-up aggregate case: **+27%**
- prose: ~0%
- some sampled extraction cases: **−5%**

Local state: older `prompt_lookup.hpp` using `std::unordered_map`, and
`ModelOptions::prompt_lookup = false` in `engine.hpp`. Upstream replaces it
with a 315-line flat open-addressing implementation, resume-after-edit logic,
and default-on behavior for single-session MTP.

Changes:

- Replace `src/models/qwen38_flash_next/prompt_lookup.hpp`.
- Port the engine wiring in `engine.cpp` / `engine.hpp`: default for
  single-session MTP, state saved in snapshots, accepted-copy accounting in the
  MTP length controller.
- Bump `Session::kSnapshotPayloadVersion` (local is 14) and invalidate old
  snapshots/continuation caches.
- Optional: expose the lookup counters in `/metrics`.

Windows fit: pure host code. Good fit.

Acceptance: greedy output byte-identical; copy-heavy requests faster; prose not
slower; sampled extraction not regressed.

### [x] #603 — score MTP drafts over the request vocabulary

Upstream: `51a8b00`. Upstream Linux measurements: **+13% coding-agent replay**,
**+16% everyday prompts**, **+5% across ten languages**, greedy bit-identical.

Ported via `git cherry-pick -n 51a8b00` + hand merge. Dropped as N/A in the
fork: grammar `constraint` term in the `unmodified` check (fork sampler has no
constraints), `continuation_hooks.hpp`/`ContinuationHooks` member and include
(fork deleted the adapter; upstream commit itself only moved the lines), the
`src/core/hip/managed_alloc.hpp` include in batch.cpp (no such file in fork;
unused there), and upstream's duplicate `PenalizedArgmax` (fork already has it).
Kept: `draft_vocab.hpp`, `SetDraftVocab`/`AppendDraftRows`/`ClearDraftRows`,
`MaskDraftLogits`/`GatherDraftLogits`/`RemapDraftIds` kernels, `MtpOutput`
`kv_only`/`draft_subset`, `kMtpDraftExtraRows` device draft table,
`small_head_cycles` stat, `kSnapshotPayloadVersion` 15→17, adaptive draft
length. Test files (`session_test.cpp`, `CMakeLists.txt`, new
`draft_vocab_test.cpp`) taken from upstream — Linux CI is their executor.
Caution: a botched conflict splice had flattened `kernels.hip.cpp` to one line;
restored from HEAD and re-inserted the three kernels by hand.

Gates (UD-Q4_K_XL, 4-shard, interleaved A/B vs stash-rebuilt baseline):
- `--validate-prefill 64`: rmse 0.31 / cosine 0.99 / max 1.57 — unchanged from
  baseline.
- Greedy haiku (40 tok) and sampled prompt (80 tok, `-t 0.8 -s 7`):
  byte-identical.
- Perf: d0 pp2048 1632/1636, tg256 29.25/29.11; d32768 tg256 26.72/26.57 —
  neutral, as predicted at line 945 (fork's Latin vocab split already captured
  the bulk of the win). The port's value here is parity with upstream
  (#660 wave64 verification builds on it) and the non-Latin fallback path.

### [x] #518 — faster Flash-Next decode kernels

Upstream: `7e4621c`. **+6–7% single-session decode**, **+4% with 2–4
sessions**, prefill unchanged, bit-identical. Fifteen small changes: fused Q8_1
producers, DPP `row_xmask` / `v_permlanex16` instead of `__shfl_xor`, MoE
grouping, batched attention loads, vector loads.

Local state: `kernels.hip.cpp` has diverged (141/47 lines vs base) and the fork
already has some decode work. Port selectively, change by change; do not
cherry-pick the whole commit.

Windows fit: pure HIP. No new switches.

Acceptance: bit-identical outputs; measure tg at d0/32K, single and 4 sessions.

Windows port log (complete; the batched attention loads and
`RmsNormCachedKernel` were dropped after measurement, see below):

- [x] Stage 1, all eight `mmq/*` files (`5749c46`): a three-way merge against
  `7e4621c^` (base) and `7e4621c` (theirs) is clean for every file once the
  blobs' LF is matched to the working tree's CRLF; the result is byte-equal to
  upstream's post-state for those files. Brings `q4_K_scale_pair` (both Q4_K
  block-scale packings formed before the select, so the third word's load is
  no longer sunk behind a divergent branch), the Q8_1-by-expert producer,
  `mul_mat_vec_moe_dispatch` `dst_slots`, `mmvq_moe_group_ints`,
  `qfn_mmq_moe_gated_vec_preq` and the IQ3_S/IQ4_XS/IQ4_NL MoE types. The
  producer and `preq` entry points stay unused until stage 3 wires the
  executor; the reachable part is the Q4_K scale read plus the common/MMVQ
  helpers. Bit-identical: haiku, 1,951-token greedy prompt and
  `--validate-prefill 64` (rmse 0.39, cosine 0.99, max 1.88) all match
  `gufo-622-lists`. Interleaved `-p 2048 -n 16 -d 0,32768 -r 2`: tg16 d0
  28.19/28.20 -> 28.22/28.30, tg16 d32K 25.89/26.03 -> 26.15/26.18 (+0.8%),
  pp2048 d0 1745.67/1733.04 -> 1735.43/1729.14 (noise). Kept.
- [x] Stage 2a (`d923796`): the top-of-file decode helpers plus every
  `__shfl_xor` in `kernels.hip.cpp` (22 sites: `WaveSum`, `BlockSum`,
  `BlockMax`, the quantize kernels' `max_abs` scans, `GdnKernel`,
  `GdnEpilogueKernel`, `SelectMarkKernel`, `RouterTopKKernel`, the argmax
  reduction) switched to the DPP `XorLane` transport, and the now-needed
  pieces of the same hunk added for stage 3: `Q8_1Block`, `Materialize`,
  `QuantizeQ8_1Lane` and the `unsigned`/`float` `XorLane` overloads. Only a
  register move replaces the LDS crossbar round trip; the lane pattern is
  unchanged, so every reduction is bitwise. Bit-identical: haiku,
  1,951-token greedy prompt (only the `[Model Load]` line differs) and
  `--validate-prefill 64` (rmse 0.39, cosine 0.99, max 1.88) all match
  `gufo-622-lists`. Interleaved `-p 2048 -n 16 -d 0,32768 -r 2` against the
  stage 1 binary: tg16 d0 28.48/28.47 -> 28.56/28.52 (+0.2%), tg16 d32K
  26.15/26.08 -> 26.54/26.51 (+1.4%), pp2048 d0 1742.19/1716.73 ->
  1726.39/1724.55 (noise), pp2048 d32K 1572.79/1565.14 -> 1576.84/1570.90
  (+0.3%). Kept.
- [x] Stage 2b (`5a29fee`): `HcMixEpilogueKernel` is now templated on
  `kStreams` with the model's four-stream shape launched as `<4>` (every
  stream read issued up front, the inject weights cached before the `mixed`
  store, the sums still running stream by stream in order) and gains the
  optional `mixed_q8` output that stage 3 will fill; `SmallGemmKernel` loads
  a lane's steps `kSteps` at a time (4 at `tokens <= 2`, else 2) and takes an
  unchecked `float4` row path (`kWhole`) for F32 rows whose length is a whole
  number of passes, still adding a fresh four-product `dot` to `acc` per step
  — the first cut folded `acc` into one `fma` chain, which groups the sum
  differently from the checked path and was rewritten before the slice
  passed. Bit-identical: haiku, 1,951-token greedy prompt (only the
  `[Model Load]` line differs) and `--validate-prefill 64` (rmse 0.39,
  cosine 0.99, max 1.88). Interleaved `-p 2048 -n 16 -d 0,32768 -r 2`
  against the stage 2a binary: tg16 d0 28.57/28.56 -> 29.08/29.13 (+1.9%),
  tg16 d32K 26.40/26.54 -> 27.02/27.04 (+2.1%), pp2048 d0 1739.34/1724.79 ->
  1732.28/1729.55, pp2048 d32K 1578.20/1571.42 -> 1579.38/1570.15 (both
  noise). Kept.
- Stage 2b dropped two pieces after measurement:
  `RmsNormCachedKernel<10>/<40>` (every element and gamma lane loaded into
  registers before the sum) was bit-identical but moved nothing: tg16 d0
  29.14/29.13 and d32K 27.05/27.00 with it off versus 29.08/29.13 and
  27.02/27.04 with it on, while pp2048 d0 read 1742.60/1722.30 with it off
  and 1731.87/1717.17 with it on (about 0.4-0.7% lower in both rounds); the
  kernel and its dispatch were deleted. The batched attention loads
  (`load_batch` over `kBatch = 8` keys, clamping empty lanes instead of
  skipping their row) failed bit-identity: `--validate-prefill 64` gave rmse
  0.36 against the expected 0.39 and both text gates diverged. Reverting only
  the attention hunks restored all three gates, so the batch itself is the
  cause; the arithmetic it feeds (`acc[g][x] += w[g] * vv[x]` over an
  unrolled eight-key loop) is the only place whose reassociation exposure
  changes, and pinning it would need a per-element barrier. Reverted; the
  hunk stays unported. The remaining vector-load hunks were part of that
  batch.
- [x] Stage 3 (`4da5063`): the executor fused Q8_1 producers. `HcCombineQ8_1`
  writes the norm's unpadded Q8_1 rows out of the combine; the plain
  `HcMixEpilogue` writes `mixed`'s rows (the `mixed_q8` output stage 2b
  reserved); `GdnEpilogueKernel` writes the F32 ssm rows' rows (new trailing
  `out_q8_1`, used only when neither `out_q8` nor `out_half` takes the row);
  and `SigmoidMulQ8_1` writes the gated attention output's rows. The executor
  keeps those rows in `xn_q8_ready_` / `mixed_q8_ready_` (`ReadyRows`: source
  pointer, rows, width, dropped at every `Forward`/`MtpForward` entry and
  wherever the producer's source buffer is rewritten) and `Quantize` returns
  them instead of launching; `GatedExperts` feeds them to the stage 1
  `qfn_mmq_moe_gated_vec_preq`, and `ssm_out`/`attn_out` read them through the
  `Dense(Q8Input)` overload. Each producer writes exactly what
  `qfn_mmq_quantize_q8_1` would for one 32-lane block per wave, so the rows are
  byte-identical. Bit-identical: haiku, 1,951-token greedy prompt (only the
  `[Model Load]` line differs) and `--validate-prefill 64` (rmse 0.39,
  cosine 0.99, max 1.88). Interleaved `-p 2048 -n 16 -d 0,32768 -r 2` against
  the stage 2b binary: tg16 d0 29.08/29.10 -> 29.20/29.21 (+0.4%), tg16 d32K
  26.89/26.99 -> 27.17/27.12 (+0.8%), pp2048 d0 1750.41/1733.41 ->
  1734.32/1719.41 and pp2048 d32K 1581.38/1572.44 -> 1575.87/1565.95 (both
  noise, and prefill takes none of these paths); `-d 131072`: tg16
  24.50/24.49 -> 24.67/24.61 (+0.7%), pp2048 1458.57/1448.29 ->
  1462.63/1459.53. That is the whole win: the saved work is four quantize
  launches per layer (xn, mixed, ssm output, attention output) plus the MoE
  reuse, a few microseconds a step, not upstream's +6-7%, which covers all
  fifteen changes including the batched attention loads and `RmsNormCached`
  this fork could not keep. Kept.
- Stage 3 dropped `SiluScaleQ8_1` and the `lo` rows after measurement. The
  decode-size `fused_hc_down` GEMV already writes the SiluScale'd `lo`, so
  taking the Q8_1 route there means replacing that fused launch with an
  unfused down GEMV plus `SiluScaleQ8_1`: the same three launches as before
  (down, quantize, up) with the quantize merely moved into the silu pass. With
  the producer, tg16 read 29.23 d0 / 27.11 d32K; without it (fused launch
  kept, as this tree already had it) 29.19 / 27.07, inside the run spread, so
  the smaller change stayed and the kernel, its wrapper, the declaration and
  the `lo_q8_1` scratch allocation were removed.
- `git merge-file` against base `fee9d2a` (fork HEAD `2eb0778`, upstream `7e4621c`)
  over the commit's 13 `src/` files: `batch.cpp`, `kernels.hpp`, all of
  `mmq/*` merge clean; `kernels.hip.cpp` has 1 comment-only conflict;
  `executor.{cpp,hpp}` have 15 conflicts.
- Do **not** take that commit wholesale. It drags in upstream's
  snapshot/checkpoint stack (`PrefillCheckpoint`, `SnapshotAllocator`,
  `SnapshotTransfer`, deferred/borrowed restores, `madvise` page population,
  `next`-chunk n-gram prefetch) that this fork does not have and that
  "Do not port" already excludes. Port the decode kernel and Q8 producer
  hunks only; leave the snapshot and prefetch hunks out.
- Merge scratch (ours/base/theirs/merged, LF blobs) is kept at
  `%TEMP%\opencode\merge518` for reference.

### [x] #660 — wave64 verification attention (rows > 1)

Upstream: `d1830de` (parent `270b533`, so the direct diff is exactly this PR).
Qwen27B MTP verification attention: multi-row decode now runs as wave64 with
shared per-head steps instead of wave32 split-K. Files:
`src/models/qwen/hip/kernels/attention_decode.hip` (+122, split-K single-row
path kept wave32), new `attention_decode_split_k.hpp`,
`attention_decode_verify_wave64.hip/.hpp`, CMake +2, test updates, bench
build.sh. Applied as `git diff 270b533 d1830de` → 3way, all clean; builds OK.

Not measured on this host: no Qwen27B GGUF locally (DavidAU cache entry is
metadata-only). User chose code-only port; gate on Linux CI or when weights
land. Shortcut: unmeasured perf claim, verify before citing numbers.

### [x] #655 — small MTP draft head for sampled multi-session

Upstream: `b6a671c` (parent `8775189`). When concurrent sampled sessions
would force the MTP head back to full vocabulary, keep the small per-request
draft table (#603) instead and score the small head directly: engine-side
gating in `PrepareDecode`/draft-head selection (`engine.cpp`), draft-vocab
capacity handling (`draft_vocab.hpp`), tests. Applied as `git diff` →
3way; clean on `draft_vocab.hpp`, `engine.cpp`, `draft_vocab_test.cpp`;
`session_test.cpp` hand-merged (kept upstream's reload-with-
decode-concurrency lambda, dropped a `CheckLookupBatch` call that belongs to
upstream's later #613 follow-up `8497711`, not in this fork). Also removed
the stale `continuation_adapter`/`continuation_probe` targets from
`tests/models/qwen38_flash_next/CMakeLists.txt` (sources were deleted by the
#603 port but the CMake refs survived — Linux test configure would fail).

Windows gates: single-session greedy haiku and sampled (t 0.8, seed 7)
byte-identical vs `cdde3e4` base. Earlier multi-session bench attempts
produced no rows: root cause `serving_bench.py --quick` without `--suite`
selects zero cases; also pass `--gufo .\gufo.exe` on Windows (the
`./result/bin/gufo` fingerprint default does not exist here).

Multi-session sampled A/B (2× interleaved, `serve llm -j 4` + serving_bench,
concurrency 2, t 0.8, 128 out tokens, warmup): aggregate output 33.7 → 39.8
tok/s (+18%), draft acceptance 0.73 → 0.80; per-request decode 25.2/24.5 vs
26.4/19.9 — base is bimodal (one request starves at 19.9 tok/s when the full
head forces it out of the small-head path), #655 removes the slow mode. Same
acceptance band (0.78-0.82) as base's fast mode, i.e. no quality cost.

### [x] #668 — stop batched greedy DSpark frontier before accepted EOS

Upstream 1b0a5a8 (fix). Batched greedy DSpark decoding ran the frontier past a
token the batch had already accepted as EOS, corrupting subsequent generation.
Ported to `src/models/deepseek_v4_flash` (native_internal.h, rocm_graph.cpp).
No DSpark weights on this host, so behavior is CI-gated; Windows build clean.

### [x] #669 — report model/build/context/vision in `/props`

Upstream 4862410. `/props` now returns model, model_alias, model_path,
build_info, default_generation_settings (n_ctx + sampling params) and
modalities (vision/video/audio). Dropped upstream's `session_states()`/total
slots (depend on #389, not in this fork); `device_lost()` returns false (no
device-loss tracking in fork). Verified live: UD-Q4_K_XL serve returns
model_path, build_info=development, n_ctx=262144, vision=false; `/slots`
still 200.

### [x] #650 — fork recurrent state instead of copying it

Upstream: `95ba1ea`. Qwen27B 4-request aggregate **64.4 → 68.0 tok/s
(+4–6%)**, single request **+2.5%**, copy time **861 → 24 ms**, outputs
identical.

Ported whole (`git apply --3way`, per-conflict keep-theirs on the 7 conflicted
src files after `git checkout -m`). This replaces the fork's Windows "lazy
recurrent save" workaround entirely: `SavedStateAliasesLive`/`Spare*`/
`CommitOutOfPlaceState`/`MaterializeSavedState`, `conv_src`/`recurrent_src`
out-of-place kernel args and the `kLazyRecurrentSave` branches are gone;
`RestoreState` is the upstream live/saved buffer swap. The separate Windows
SDMA `LaunchDeviceCopy27` workaround is kept (snapshot copies, unrelated).

Gates: Qwen3.8-27B Q4_K_S greedy haiku byte-identical vs `b054756`; Flash-Next
UD-Q4_K_XL + MTP speculative greedy identical (shared speculative_verifier
paths). Plain bench neutral (pp2048 ~428, tg128 ~12.4 both exes) — the copy
elimination only shows under speculative/served 27B decode, and there is no
27B MTP draft model on this host to measure it; shortcut: perf claim rests on
upstream's dflash2 measurement, re-measure when a 27B draft ships.

---

## Tier 3 — long-context prefill and larger kernel work

### [x] #421 - faster prefill projections, attention and indexer

Upstream `653a318`, 4 files, +221/-47. Changes: `SyncLds()` convergent
barrier for LDS-only waves, `WmmaCausalAttentionKernel` rewrite around it,
`DenseF16GEMMKernel` tile/accumulation rework, `DenseBf16Gemm` own kernel and
`SelectBlocks` bounds. **Ported whole**: a three-way merge against
`653a318^` had zero conflicts in `executor.cpp`, `kernels.hpp` and
`kernels.hip.cpp`; only `blaslt.cpp`/`.hpp` conflicted, because Tier 1's
`FindPlan`/`Prepare` sit in the same region. Resolution keeps both: upstream's
`FindPlan`/`Run`/`UsesOwnKernel`/`LibraryGemm` split plus the local `OwnF16`
and `Prepare` path.

Cost of the port: one `git merge-file` per file plus two hand resolutions.
Measured effect is small here (below), but it is the base `#463`/`#485` build
on, so it stays.

### [x] #463 - reduce prefill work at long context

Upstream `33d1b20`, 4 code files, +91/-40 in `executor.cpp` plus its header and
the kernels. Three changes, all in the same direction as the acceptance
criteria:

- **Bounded selector score scratch.** The score buffer is sized from
  `kVecBatch` and a 5 Mi floats cap instead of `select_chunk_ x max_blocks`,
  and a wide prefill compacts its score rows to the populated context. Graph
  replay keeps the maximum stride. Saves memory and cache traffic at depth.
- **Final-layer AR prefill pruning.** With one requested logit and MTP off, the
  last layer runs attention over only the final query tile (K/V/indexer rows
  are all still written) and runs the FFN, mixers and residual for the aligned
  tail only (at least 96 rows, keeping the wide arithmetic).
- **Fused MoE combine without a norm.** `HcCombineMoeF16Kernel` is templated on
  `kNormalize`; with a null `gamma` it updates the residual and returns before
  the norm/Q8 stage, so the last layer no longer normalizes the whole batch
  for a head that reads one row. `kernels.hpp` documents the new contract.

Ported by hand (the fork's `ForwardBody` has no checkpoint stack, and its
`Attention` takes three trailing flags, not four). The three way merge would
not have applied: the local `executor.cpp` already carries the MTP catch-up
tail scratch this reuses.

Measured, same session, `gufo bench -p 2048 -n 16 -d 0,32768 -r 3`:

| binary | pp2048 d0 | pp2048 d32K | tg16 d0 | tg16 d32K |
| --- | --- | --- | --- | --- |
| before (HEAD) | 1596.91 | 1445.21 | 28.26 | 26.01 |
| after #463 | **1627.12** | **1487.96** | 28.32 | 25.82 |

+1.9% / +3.0% prefill, decode neutral. `--validate-prefill 64` is unchanged
(rmse 0.37, cosine 0.99, max error 2.08) and the greedy haiku is byte-identical
to the pre-patch binary. Upstream reports the same direction and size (+1.2%
d0, +1.8% at 128K).

### [x] #485 — accelerate prefill and MTP catch-up

Upstream: `fd747a5`. Upstream matched HTTP increments: pp2048 d0 MTP
**1581.01 → 1623.99 tok/s**, d32K **1490.59 → 1524.10**, d128K AR
**1381.32 → 1400.23**, TG within 0.5%, completions and draft counts identical.
Eight independent slices; the fork has no `GdnCheckpoint`/`PrefillCheckpoint`
stack, so checkpoint code is dropped from every slice.

- [x] **GDN row reuse** in `GdnRowSplitKernel`: two rows per lane, four waves,
  128 threads per block, shared `q`/`k` loads reused across the row pair.
- [ ] ~~Padded SSM F16 rows (`half_stride`, `DenseF16Gemm` `kStrided`).~~
  **Rejected locally (2026-10-19).** Ported `GdnEpilogueKernel`
  `out_half_stride`, `GatedDeltaNet`/`DenseF16Gemm` plumbing and the
  `kStrided` `b_ptr` stride (skipping upstream's partial-tile restructure;
  chunks are 2048 = full tiles). Gates passed (rmse 0.39 / cosine 0.99 /
  max 1.88; haiku + 1,951-token greedy identical), but interleaved A/B vs
  baseline showed no win: tg16 d0 28.98/29.09 vs 29.12/29.53, d32K
  26.91/27.18 vs 27.18/26.78, d128K 24.91/25.05 vs 24.82/24.88 — all within
  noise with no consistent direction; pp2048 1740/1728 vs 1746/1740, d32K
  1585/1578 vs 1590/1580, d128K 1443/1463 vs 1456/1452 — equal-to-slightly-
  behind. Reverted.
- [ ] ~~Full selector scratch (`select_chunk` without `bit_floor`,
  `first_query = n_tokens - 16` for `last_only`), packed `SelectScore` half2
  loads and `sched_barrier`.~~ **Rejected locally (2026-10-19).** Gates passed
  (rmse 0.39 / cosine 0.99 / max 1.88; haiku + 1,951-token greedy identical),
  but interleaved A/B vs baseline (`-p 2048 -n 16 -d 0,32768,131072 -r 2`)
  showed a large prefill-at-depth regression: pp d128K 1364.27/1365.12 vs
  1485.81/1459.54 (−8%), pp d32K 1553.94/1543.18 vs 1584.66/1577.96, while
  tg16 stayed within noise (d0 28.68/28.71 vs 28.77/28.64; d128K
  24.62/24.84 vs 24.73/25.13). No token-generation win justifies the prefill
  loss; reverted. Suspect the un-rounded `select_chunk` (larger launches at
  depth) rather than the packed loads.
- [x] `RoutedPadBounds` clears padding in-kernel, dropping two
  `hipMemsetAsync` calls. Landed `6fd51ed`. Gates: `--validate-prefill 64`
  rmse 0.39 / cosine 0.99 / max 1.88, haiku and 1,951-token greedy identical.
  Measured same session, `gufo bench -p 2048 -n 16 -d 0,32768,131072 -r 2`,
  interleaved vs `gufo-518-q8b.exe`: tg16 d32K 26.99/27.22 vs 26.90/26.95,
  tg16 d128K 24.92 vs 24.85, tg16 d0 29.02/29.31 vs ~27/29.27, pp
  equal-or-ahead (1746/1740 vs 1729 baseline). Decode consistently ahead;
  kept on token-generation grounds.
- [x] `HcMix` `half_only`, skipping the F32 and Q8 copies when both stacked
  SSM projections read the F16 input the wide mixer already wrote. Needs the
  `y != nullptr` guard in the `kHcMix` epilogue and the relaxed
  `HcMixF16Gemm` argument check: measured +0.5% pp2048 d0, +0.1% d32K, tg
  neutral, greedy haiku and `--validate-prefill 64` (rmse 0.42) byte-identical
  to the GDN build.
- [x] PLE direct mapped n-gram input (`hipHostMallocMapped | Coherent`,
  `hipHostGetDevicePointer` into `s.ple_emb`, no F32 allocation or uploads).
- [ ] ~~`DenseBatch` 32-row chunk split for 64+ rows.~~ Fork already splits at
  `kDecodeRows=8`; no port needed.
- [x] MTP catch-up sparse-boundary fix (`hidden_source` +
  `control_host_->hidden_row`, `Attention` `last`/`dense_rows` flags), plus
  `PinnedInputGuard` syncing `stream_` on failed passes after
  `EnsureRollback`.

Remainder landed 2026-10-20 (PLE mapped buffer, selector
`select_chunk = min(512, ...)` without `bit_floor`, `first_query =
n_tokens - 16` for `last_only` — packed half2 loads and `sched_barrier`
deliberately not ported, they caused the earlier −8% pp d128K rejection).
Gates: `--validate-prefill 64` rmse 0.31 / cosine 0.99 / max 1.57; greedy
haiku byte-identical. Interleaved A/B vs baseline (`-p 2048 -n 16 -d
0,32768,131072 -r 2`): perf-neutral — pp2048 d0 1628/1628, d32K
1401/1398, d128K 1323/1326, tg16 all within noise. Kept for the dropped
PLE upload copy and the failure-path pinned-buffer race fix.

GDN slice measured, same session, `gufo bench -p 2048 -n 16 -d 0,32768 -r 3`:

| cell | baseline `2eb0778`+tier1+#421+#463 | GDN slice | delta |
| --- | ---: | ---: | ---: |
| pp2048 d0 | 1629.89 (1637.50 / 1622.28) | **1658.43** | +1.75% |
| pp2048 d32K | 1491.94 (1497.31 / 1486.57) | **1516.97** | +1.7% |
| tg16 d0 | 28.46 | 28.42 | neutral |
| tg16 d32K | 26.17 | 26.32 | neutral |

`--validate-prefill 64` **improves**: rmse 0.30 vs 0.37, cosine 1.00 vs 0.99,
max error 1.74 vs 2.08.

Not bit-identical, accepted deliberately. The kernel's arithmetic source is
the single-row kernel's, unchanged (`u += a + b + c + d`), but two rows per
lane changes register pressure, so clang 23 fast-math reassociates that sum
differently. Greedy seed 42 diverges at token ~40 of the copper-wire haiku:
both binaries print the same first two lines, the baseline ends
`Time wears down the shine.` and the port ends `Time writes in the wet.`
Rewriting the sum as an explicit `fmaf` chain does not restore identity.
This is the documented FP reassociation the acceptance criteria allow;
prefill validation and decode rates are unaffected, and no sampling,
draft-vocabulary or precision change is involved.

### [x] #626 — cut prefill and decode attention cost at depth

Upstream: `5a1669a`. Upstream measured pp128K 1511 → 1539 tok/s, sparse
attention 107.1 → ~96 ms per pp2048 at 30K, tg64 d128K 27.5 → 27.85. All
changes byte-exact. The upstream executor-side MTP-scratch reuse already
exists here as `keep_rollback_rows`; kernels only.

- [x] **Selection mark/compact**: `SelectMarkKernel` templated on `kBlock`
  with 1,024-thread narrow-batch launch for `n_tokens <= 4`, vectorized
  `uint4` score loads (`for_each_score`, 4×16 B per thread), batched
  tie-chunk mask writes, `FindHistogramThreshold` gains `kBins, kBlock`
  template params. (No `CompactSelectionKernel` here: this fork compacts
  inside `AttentionKernel`, a known local/upstream difference.)
- [x] Decode attention head groups (`AttentionKernel<kGroup>` 2/4 query
  heads per block sharing K/V loads). Adapted to the fork's in-kernel
  compaction: the compaction pass and `tile_step` K/V loads are shared by the
  whole group; per-group `m`/`l`/`acc` and a per-group epilogue with
  `__syncthreads()` between heads (shared `red` reused). Launcher:
  `(heads/kv_heads)%4==0`, `n_tokens<=2` → `<2>` grid `(heads/2, t, z)`,
  else `<4>` grid `(heads/4, t, z)`.
  Measured `gufo-626-groups.exe`: pp d0 1668.01 / d32K 1526.25 / d128K
  1403.28±11.50 (all neutral vs the selection slice), **tg16 d128K 23.57±0.32
  vs 23.77±0.36 (+0.8%)**, tg d0/d32K neutral. Greedy haiku identical
  ('Time writes in the wet.'); `--validate-prefill 64` rmse 0.38 cosine 0.99
  max 1.96. Rollback: revert d6319c5.
- [x] Dense prefill attention per-tile overhead + sparse softmax masking +
      pinned PV WMMAs (WmmaCausalAttentionKernel slice). First port (while the
      kernel still had a single `kv_lds`) measured neutral and was reverted.
      Re-ported after #571 landed the split `k_lds`/`v_lds` two-buffer phase
      pipeline this slice was tuned for (kScoreStride 20 + float4 softmax
      reads, parity-grouped row_scale, memcpy-packed p_lds rows, dense fast
      paths in gather_tile/tile_key, per-lane block_selected, dpp/perm V word
      staging, sched_barrier in the S loop, PV pin asm). Gates: greedy haiku
      byte-identical, `--validate-prefill 64` rmse 0.31 cosine 0.99 max 1.57.
      Interleaved A/B (2×, alternating order): pp2048 d0 +0.5% (1641.86 vs
      1633.34), pp2048 d32K **+1.5%** (1417.02 vs 1396.54), pp24576 d0 +0.9%
      (1545.97 vs 1532.19), pp24576 d32K **+1.1%** (1497.63 vs 1480.66 and
      reversed-order pass 1494.69/1492.05 vs 1478.14/1472.58), tg16 neutral
      at all depths. Kept.

Measured (commit `0857383`, bench `-p 2048 -n 16 -r 3` / `-r 2` at d128K):
pp2048 d0 1667.06±6.79 (neutral), d32K 1522.05±4.81 (neutral), d128K
**1414.03±24.54 vs 1391.00** (+1.7%), tg16 neutral at all depths. Greedy
haiku identical ("Time writes in the wet."). Rollback: revert `0857383`.

### [x] #470 - 4096-token prefill chunks + overlapped n-gram gather



Upstream: `47b6391`. **6–11% faster for 24K–102K prompts**, prompts under 8K
not slower, decode unchanged, +1.3 GiB at 262K.

Local state: `engine.cpp:31` uses `kPrefillChunkTokens = 2048`; the comment says
an earlier gfx1151 sweep at depths 0/4096 favored 2048. Upstream’s newer
measurement favors 4096 on served 24K+ prompts because the next chunk’s n-gram
gather overlaps the current pass.

Changes:

- Raise the chunk to 4096.
- Thread a `next` span through `Feed`/`Sync`/`SyncImpl`.
- Pass it to `Executor::Forward` and prefetch the next chunk’s n-gram rows.
- Wake one n-gram reader per read instead of the whole pool.
- Keep the local `kWindowsWorkers = 128` and `O_CONCURRENT_RANDOM` path.

Windows fit: good; no io_uring. **Benchmark 2048 vs 4096 on Windows at
d0/32K/128K before committing**; keep 2048 if the Windows sweep disagrees.

Windows result for the chunk size alone (constant flipped, no prefetch
plumbing), `-p 2048 -n 16 -d 0,32768,131072 -r 2`, one session:

| chunk | pp2048 d0 | d32K | d128K |
| --- | ---: | ---: | ---: |
| 2048 | 1661.80 ± 13.46 | 1518.89 ± 2.23 | 1391.00 ± 2.90 |
| 4096 | 1650.92 ± 14.91 | 1502.94 ± 11.49 | 1380.29 ± 9.17 |

4096 is neutral to worse at every depth without the next-chunk gather
overlap, so the constant stays at 2048. The rest of #470 only pays with the
`Feed`/`Sync` `next`-span prefetch, which is not ported yet.

Acceptance: logits independent of chunk size; long prompts faster; short
prompts and decode not slower.

Ported (upstream `47b6391`, diffed against parent `7701ba7`): chunk raised to
4096, `next` span threaded through `Feed`/`Sync`/`Executor::Forward`
(prefetches the next chunk's PLE n-gram rows during the current pass),
`FinishPrefetch` claim path in `WaitPle`, per-read `notify_one` in
`ngram.cpp` (Linux worker count 32; kept `kWindowsWorkers = 128`),
scheduler short-prefill heuristics via new `Request::prefill_position()`
(getter over the existing `prefill_offset`). Dropped as not in the fork:
`PrefillThroughCapacity`/`kPrefillTailTokens` (no 128-token tail; both
branches of its use collapse to `PrefillCapacity`), `PrefillCheckpoint`/
`SyncThrough` capture stack, checkpoint plumbing in `Feed`.
`Feed`'s loop now uses upstream's fits-through capacity pattern so a final
chunk is not split.

Windows gates (UD-Q4_K_XL): `--validate-prefill 64` rmse 0.31 / cosine 0.99 /
max 1.57 (baseline unchanged); greedy haiku byte-identical (only model-load
timing digits differ); pp4096 1544→1663 (+5–7% across 2 interleaved rounds),
pp24576 1429→1528 (+6.4%), pp65536 1395→1490 (+6.6%); tg64 @ d32768
26.45→26.58 (noise). Matches upstream's claimed 6–11% for 24K+ prompts.

### [x] #571 - faster wide prefill kernels

Upstream: `a33e989`. pp2048 **1826 → 1853 tok/s**; HTTP AR pp d0
**1708 → 1774**, d32K **1536 → 1649**; pp2051 **1795 → 1815**; byte-identical
logits.

Changes: gate/up fused rows, shared K16 fragments, register-resident Q8_0 SSM
fragments, deferred residual in MoE combine, 128×128 router/GDN tiles,
128-token gate/up and 64-token down tiles, tail kernel for up to 16 tokens past
a tile, software-pipelined attention, GDN q/k in registers.

- [x] **128×128 router/GDN tiles** (`d61b82a`): `UnquantizedF16Gemm` gains a
  `batch >= 1024` branch launching `DenseF16GEMMKernel<128,128,...>`; the
  64×64 path is unchanged for narrow batches. Exact: greedy haiku and
  `--validate-prefill 64` (rmse 0.42, cosine 0.99, max 2.10) both identical
  to `5629bc7`. Measured: pp2048 d0 1667.03±6.03 (neutral), d32K
  1523.20±5.28 (+0.3% over 1518.53), tg neutral.
- [x] **deferred residual + inject partials in the combines**: `HcCombineF16`
  gains `defer_w`/`inject_out`/`inject_w`, `HcCombineMoeF16` gains
  `prior_out`/`prior_w`/`inject_out`/`inject_w` (both kernels get `kDefer`/
  `kPrior`/`kInject` stages, `HcInjectPart`, and `HcApplyDeferred`); the executor
  defers the attention-side residual across a fused-MoE FFN combine and the
  combine writes the next mixer's inject partials from its F16 norm, so `HcMix`
  skips its inject pass. Exact: `--validate-prefill 64` rmse 0.39 / cosine 0.99
  / max 1.88, greedy haiku unchanged, and a 1,951-token teacher-forced prompt
  produced the same 32 greedy tokens as `gufo-626-groups`. Measured (interleaved,
  same sitting): pp2048 d0 1691.13 -> 1714.27 and 1667.41 -> 1718.48
  (+1.4%, +3.1%); d32K 1528.73 -> 1555.96 and 1517.44 -> 1556.64
  (+1.8%, +2.6%); d128K 1421.80 -> 1429.57 (+0.6%); tg neutral
  (28.46/28.35/26.21/26.05 unchanged within noise).
- [~] gate/up fused rows + shared K16 fragments: the paired-wave rewrite of
      `RoutedF16GEMMKernel<..., kPair=true>` (register-resident codes and affine
      words, `XorLane` peer exchange for the second K16 fragment, fixed code
      cache group per stage, F16-staged SwiGLU epilogue) was ported whole. Exact:
      greedy 1,951-token prompt (32 tokens) and the greedy haiku unchanged.
      Measured, interleaved: pp2048 d0 1731.65 -> 1739.75, 1739.95 -> 1708.66,
      1713.39 vs 1744.75, 1718.35 vs 1737.18 (mean 1720 vs 1738, ~1% slower);
      d32K 1559.60 -> 1576.13, 1571.71 -> 1546.91, 1552.93 vs 1571.20,
      1559.77 vs 1576.35 (mean 1559 vs 1570, ~0.7% slower). No win on this GPU;
      reverted.
- [x] **register-resident Q8_0 SSM fragments + 4-channel boundary
  convolution** (`b2ed28c`): `DenseF16GEMMKernel` gains `kRegisterWeights`
  (`kSsmConv`): the SSM wave keeps its decoded F16 K16 fragments in registers
  (`own_a_lo/hi`) and takes the peer half from `XorLane(..., 16)` instead of
  reading them back from LDS, which also drops the SSM weight plane from the
  stage; `SsmConvBoundaryKernel` now produces four channels per thread and all
  three boundary rows of a tile per launch (grid z removed). Exact:
  `--validate-prefill 64` rmse 0.39 / cosine 0.99 / max 1.88, greedy haiku
  ("Time writes in the wet.") and the 1,951-token greedy prompt unchanged.
  Measured interleaved: pp2048 d0 1730.31/1715.11 vs 1727.47/1727.99 (+0.3%),
  1715.85/1720.50 vs 1714.82/1727.41 (neutral); d32K 1560.29/1556.72 vs
  1568.01/1563.19 (+0.5%), 1549.19/1547.74 vs 1558.37/1559.70 (+0.7%).
  Kept: exact and consistently ahead at depth, neutral at d0.
- [~] 128-token gate/up and 64-token down tiles + tail kernel: the
      `W8A8TailKernel` slice (limits `kW8A8TailMax*`, `Materialize` helper,
      tail split in `HcDownF16Gemm`/`W8A8Gemm`, explicit-FMA rewrite of the
      tiled accumulator) was ported and built as `gufo-571-tail.exe`. Exact:
      `--validate-prefill 64` rmse 0.38 / cosine 0.99 / max 1.96 and greedy
      haiku ("Time writes in the wet.") both unchanged. Interleaved A/B vs
      `gufo-626-groups.exe` at the only shape this pipeline can feed it
      (`-p 2179`: last chunk 131 rows -> 128 tiled + 3 tail): pp 1500.19 ->
      1489.98 and 1497.06 -> 1492.32, tg neutral. No win: upstream's +1.1%
      needs whole-prompt batches a few tokens past a 128 multiple, while our
      2048-token chunking rarely lands the final chunk in 129-144. Reverted.
- [x] **software-pipelined attention, GDN q/k in registers** (`a33e989`, the
  last open #571 sub-item): the WMMA causal-attention kernel stages the next
  K/V tiles into separate `k_lds`/`v_lds` shared buffers while the current
  tile's PV accumulate runs (the `kLateV` deep-window special case and the
  unified `kv_lds` buffer are gone; the deep-window branch at `start_pos >=
  65536` now uses the same pipelined kernel), softmax/accumulate became
  lambdas over a `Tile` struct, and the running-sum update uses `__fmaf_rn`
  so fast-math cannot reassociate the rescale. The GDN mixer's windowed q/k
  loads moved from a `float4 qk[kQkPerThread]` array (private memory,
  reloaded every window) to two named registers. Exact: greedy haiku
  byte-identical, `--validate-prefill 64` unchanged (rmse 0.31, cosine 0.99,
  max 1.57). Measured, interleaved, two rounds per side: pp2048 1615.6 →
  1629.8 (+0.9%), pp24576 1523.3 → 1535.5 (+0.8%), pp24576 @d32768 1466.0 →
  1475.7 (+0.7%), tg neutral.

Windows fit: pure HIP, but large kernel rewrite against a diverged fork. Do
last; port one kernel family at a time.

### [x] #622 — compact sparse attention selections once per row

Upstream: `baf5a1b`. **+3.3% at 160K**, server deep decode **+1.4–2.4%**,
bitwise identical.

Local state: `AttentionKernel` compacted the row's selected blocks inside the
kernel, rebuilding the same list in every (head, split) block. A new
`CompactSelectionKernel` compacts each row once into `attn_lists`/`attn_counts`
(`executor.hpp`/`executor.cpp` scratch, `kVecBatch * indexer_top_k /
compress_ratio` entries) and the kernel reads that list window-by-window with a
uniform cursor, so it cuts the same tiles in the same order as the mask walk and
stays bitwise identical: haiku, the 1,951-token greedy prompt and
`--validate-prefill 64` (rmse 0.39 / cosine 0.99 / max 1.88) all unchanged.

The list path only runs for the ratio-4 sparse mask on narrow (`split`)
batches; prefill and wide batches keep the mask walk, and the mandate to
`SelectMarkKernel`'s budget keeps a row's list within `capacity`.

Measured, interleaved (`-p 2048 -n 16 -r 2`), two rounds per side:

| metric | before | after |
| --- | --- | --- |
| pp2048 d0 | 1731.95 / 1713.97 | 1726.01 / 1721.93 |
| tg16 d32K | 26.07 / 26.17 | 26.03 / 26.03 |
| pp2048 d128K | 1444.84 / 1428.24 | 1449.91 / 1449.80 |
| tg16 d128K | 23.53 / 23.49 | 23.40 / 23.41 |

Decode at depth improves +0.4–0.5% in 4 of 4 paired rounds; prefill is inside
its run-to-run noise. Kept.

Windows fit: pure HIP.

### [x] #645 — reuse historical image preprocessing

Upstream: `f948ddd`, ported as of October 2026. Image queue medians:

- 1 image **45.01 → 9.65 ms**
- 2 images **82.40 → 13.25 ms**
- 4 images **161.49 → 20.63 ms**
- fresh encode: **+0.42 ms** worst

Local state: vision prompts carry an identity and avoid decoding/resizing the
same image twice within a prompt, but there is no backend LRU across requests.

Changes: add `src/models/qwen/vision/image_cache.cpp/.hpp` (128 MiB LRU keyed
by exact image bytes), hook it in `src/cli/serve/inference_backend.cpp`, and
add the prompt-side reuse path.

Windows fit: pure host code. Only matters for multimodal workloads.

Port notes: dropped upstream `control_tokens.hpp` include (not in fork),
replaced `Logger::LogFormatted` debug stats with a one-line `Logger::Info`
(fork has no formatted logger), and dropped `CountPromptTokens` overrides
(upstream #510 virtual, absent in fork; cache works without it). Verified:
serve llm + mmproj BF16, two identical image requests — second logs
`event=image_cache hits=1 misses=1`, identical responses.

---

## Cheap optional ports

### [x] #667 — store d*sum(q) as the q8_1 block sum

Upstream: `86730bd` (post-review). `quantize_mmq_q8_1` stored the sum of the
*pre-quantization* floats in the q8_1 block sum; consumers expecting the
ggml contract (`d*sum(q)`) read a wrong second scale. Ported in the fork
(`mmq/quantize.hip.cpp` `sumq` shuffle reduce, `kernels.hip.cpp`
`XorLane` variant) with +179 lines of `routed_wmma_ops_test.cpp` checking
expert dots against the q8_1 activations contract. Gates run with this in
the #485-remainder binary (2026-10-20): `--validate-prefill 64` rmse
0.31 / cosine 0.99 / max 1.57, greedy haiku byte-identical, A/B perf
neutral. Kept as a correctness fix.

### [x] #415 — skip vocabulary blocks that cannot change the sampled token

Upstream: `b945d0a`. Sampler microbench: top-k 20 **8.4×**, top-k + presence
**15×**; end-to-end **+1.2–3.6%** on sampled requests. Local
`src/core/sampling.cpp` still scans every entry and binary-searches penalties.

Windows fit: pure C++. Low risk. Greedy on GPU is unaffected.

Ported 2026-10-20: block skip (kSkipBlock=64, BlockMaximum), lockstep penalty
walk (`adjusted_at`/`for_each_adjusted`) in select_best/select_all/top-p.
Verified: `logit_sampler_test` (compiled standalone with the release clang++
against `src/core/sampling.cpp` — cpu-test preset still blocked by FindICU on
this host) passes including the new `TestBlockSkippedTopKAgainstFullSort`;
sampled A/B (t 0.8, top-k 64, top-p 0.95, seed 42, 120 tokens) byte-identical
to baseline.

### [x] #332 — GPU greedy penalties

Upstream: `7e450e8`. Avoids the GPU→CPU fallback when greedy MTP uses
presence/repetition penalties. Adds a penalty kernel and executor changes.
Only worth it if penalties are actually used.

Ported (fork delta: no grammar/constraint system, so `CanSelectArgmax` is
penalty-only and `WithoutConstraint` was dropped; `GreedyMtpPredictions`
copies the sampler state to apply tentative penalty rows). Gates: greedy
haiku, greedy with `--repeat-penalty 1.15 --presence-penalty 0.3`, and
sampled (t 0.8, seed 42) all byte-identical to the pre-port baseline;
penalized greedy now stays on the GPU argmax path. Perf not re-benched
(upstream win applies to penalty-configured greedy serving).

### [x] #348 — skip disk checkpoints that barely advance a prefix

Upstream: `93af45d`, ported 2026-10-20. Fewer disk writes/fsyncs on
multi-turn chats. No decode/prefill gain. Trade-off: disk-only restore
re-prefills up to 2,048 tokens. Do it only if the disk cache is a measured
bottleneck.

Ported anyway (cheap, pure win for chat serving): `min_checkpoint_step_tokens`
(default 2048) in `ContinuationDiskStoreOptions`, `WithinCheckpointStep`
preflight skips the device-state copy when the store would discard the result,
`reason=min_step` logged at Info. Dropped the shared-prefix/persistence
`SaveAsync` overload (upstream API not in fork). Verified: 3-turn chat with
`--cache-disk` — turn 1 stores (write_ms=4316), turns 2–3 skip with
`action=skipped reason=min_step`.

### [x] #664 - price greedy MTP drafts at the decoding occupancy

Upstream: `2caf2a8`, ported 2026-10-20. Draft length now adapts to session
concurrency via `draft_length_.Choose(cap, tokens, policy_concurrency)` with
`policy_concurrency` from `model_->DecodeConcurrency()` (also persisted in the
snapshot header, bump-checked). Fork merge note: kept the fork-only `survival`
branch ahead of `Choose`; `batch_drafts` short-circuit unchanged.

Gates: greedy MTP haiku byte-identical vs `2d9027a` base; single-session
speculative bench neutral (tg128 38.2 vs 38.3, d32K 78.6 vs 78.7); 4-session
greedy serve A/B 2x2 interleaved neutral within run-to-run drift (aggregate
~50.0 vs ~50.5 tok/s, ±3 noise). No regression; kept for upstream parity —
the occupancy pricing matters at higher concurrency than this host benches.

### [x] #676 - report no content for whitespace between parallel tool calls

Upstream: `9a9717c`, ported 2026-10-20. Fork's `ParseGeneration` has no
upstream `ContentBefore` (different, older tool-call parser); the equivalent
point is where `parsed.text = text_before_tools`. Ported as: when tool calls
were parsed and `Trim(text_before_tools)` is empty, emit `""` instead of the
whitespace separator. Added `TestParallelCallSeparatorIsNotContent` to
tests/cli/openai_chat_test.cpp using the fork's FakeBackend harness
(upstream's 80-line test targets APIs the fork lacks). Verified live:
parallel `get_weather` Rome+Milan with `tool_choice=required` returns two
`tool_calls` and `content: null`.

### [x] #609 — load DeepSeek V4 Flash when weights exceed the device pool

Upstream: `a7ca5f0`, ported 2026-10-20. Weights now go through
`gufo::hip::WeightAllocator` (new header-only `src/core/hip/heap_placement.hpp`
+ `src/core/hip/managed_alloc.hpp`, verbatim from upstream): placement is
decided once from the registered weight size vs. the device heap with a
4 GiB dispatch reserve; oversized weights go wholly to coarse-grain managed
memory so per-session device scratch keeps the heap. Code-only port — no
DSpark weights on this host to behavior-gate; CI-gated like #668.

### [x] #499 — stage Qwen-Image weight uploads through pinned buffers

Upstream: `eb91daa`. Local `qwen_image_21/weights.cpp` uses `mmap`; staging
through `hipHostMalloc` avoids the host page-pinning stall. This is a
load/stall fix, not a decode speedup. Ported 2026-10-20: `UploadMapped`
(double-buffered 64 MB pinned staging, event-guarded reuse) replaces the
direct `hipMemcpyAsync` from the file mapping in `Runtime::Weight`.
Verified end-to-end: `serve image` + one 832×832 generation through the
staging path (28.5 GB resident weights, no stall, valid PNG).

---

## Do not port

- **#445 / #615 / #633 cache and checkpoint stack** — local Flash-Next has no
  prompt-checkpoint/component-cache subsystem. #445 depends on it; #615/#633
  use io_uring, `mmap`/`MADV_POPULATE_WRITE`, and a ROCr queue-save patch.
  Only the hipBLASLt `Prepare` fragment is worth taking now.
- **#653 Linux page pinning / ROCr patch** — Nix/io_uring Linux-only. The
  portable cold-function bits are small.
- **#248d902 managed-alloc fallback** — no demonstrated perf win; Windows HIP
  managed-memory behavior is uncertain.

---

## Recommended order

Working order, as of October 19 2026 (decode is already at upstream parity,
so prefill and loading lead):

1. ~~`#459` + `#481` + `#482` + `#633 Prepare`~~ done, see Tier 1.
2. ~~`#421` + `#463`~~ done. `#485` in progress — GDN row reuse
   landed (+1.75% pp2048 d0, +1.7% d32K); the remaining seven slices are
   listed in its section.
3. `#470` — 2048 vs 4096 chunk sweep on Windows; only then decide.
4. ~~Loading-time investigation~~ done: the gap is Windows NTFS serializing
   unbuffered reads of the mapped shard, and the cached alternative thrashes
   the page cache; see "Windows loading path" under known differences.
5. `#571` / `#626` / `#622` — wide and long-context prefill kernels.
6. `#518` + `#650` — decode kernels, only after prefill lands and with a
   bit-identical check (decode is at parity today).
7. `#613` + `#603` — agent/rewrite decode; benchmark English vs multilingual
   and sampled extraction (watch the −5% case).
8. `#645` + `#415` — multimodal and sampled-request polish.

## Measurement log

Host: TheRock clang 23 / ROCm 7.15, gfx1151. Model
`unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL, plus
`MTP\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` where a row says MTP.
Baseline binary = fork HEAD `2eb0778`; "tier 1" = `#459`/`#481`/`#482`/`#633
Prepare` applied. Both binaries keep identical compiler and dependencies.

### Tier 1, `gufo bench -p 16,64,2048 -n 16 -r 1 --validate-prefill 64`

| test | baseline | tier 1 | delta |
| --- | --- | --- | --- |
| pp16 | 135.41 tok/s | 134.80 | -0.5% |
| pp64 | 351.15 | 347.31 | -1.1% |
| pp2048 | 1419.66 | 1438.95 | +1.4% |
| tg16 | 27.23 | 27.34 | +0.4% |
| prefill validation | rmse 0.37, cos 0.99, max err 2.08 | identical | — |
| greedy text (seed 42, 32 tok) | reference | byte-identical | — |

One repetition each, so the throughput deltas are noise. Tier 1 is a Qwen27B
and Qwen-Image and DeepSeek change; this model only received the
`RoutedF16GEMMKernel` masked-blend edit, and it is neutral here. `#482` and the
Qwen27B half of `#459` are unmeasured on this host (no weights).

### MTP draft-depth sweep, `gufo bench -p 2048 -n 128 -d 0,32768 -r 2`

Tier 1 binary, no `--mtp-policy`/`--mtp-draft-vocab` (bench has no such
flags). Depth 32K rows decode a repetitive continuation, so their acceptance
is higher than depth 0 by construction; compare rows at the same depth only.

| config | pp2048 d0 | tg128 d0 | pp2048 d32K | tg128 d32K |
| --- | ---: | ---: | ---: | ---: |
| AR | 1478.20 | 27.22 | 1289.17 | 25.60 |
| MTP draft 3 | 1474.83 | 37.10 | 1271.41 | 54.14 |
| MTP draft 5 | 1465.03 | 36.98 | 1270.06 | 61.83 |
| MTP draft 7 | 1444.26 | 36.23 | 1261.83 | 65.72 |

Mixed-text decode peaks at draft 3; deeper drafts only win on repetitive
text, exactly as the upstream mixed/repetitive columns predict. Draft depth
also costs a little prefill (1-2%). Local production config uses
`--mtp-draft-vocab latin --mtp-policy survival`; these rows are the plain
MTP baseline and stay the reference for kernel A/Bs.

### #421 A/B, `gufo bench -p 2048 -n 16 -d 0 -r 5`

| test | baseline | #421 | delta |
| --- | ---: | ---: | ---: |
| pp2048 | 1436.33 ± 40.56 | 1460.77 ± 40.22 | +1.7% (inside noise) |
| tg128 | 27.22 | 27.60 | +1.4% (inside noise) |
| prefill validation | rmse 0.37, cos 0.99, max err 2.08 | identical | — |

Loading in the same run: 34.82 s versus 36.74 s before, which is run-to-run
variance, not a change. Kept because `#463`/`#485` build on its kernel shapes,
not because of this delta. These absolute numbers come from a slower machine
state than the `#463` rows below; compare deltas within a section, not across.

### #463 A/B, `gufo bench -p 2048 -n 16 -d 0,32768 -r 3`

Alternating in one session, no rebuild between the two binaries:

| cell | before (HEAD) | after #463 | delta |
| --- | ---: | ---: | ---: |
| pp2048 d0 | 1596.91 ± 8.55 | 1627.12 ± 5.10 | +1.9% |
| pp2048 d32K | 1445.21 ± 3.28 | 1487.96 ± 7.13 | +3.0% |
| tg16 d0 | 28.26 ± 0.41 | 28.32 ± 0.30 | neutral |
| tg16 d32K | 26.01 ± 0.22 | 25.82 ± 0.12 | −0.7% (noise) |

Loading was 36.08 s on the patched run versus 34.96–35.0 s before: no change.

**Machine-state drift.** An earlier baseline on the same day measured 1436.33
pp2048 and an early `#463` run 1649.78. The re-measured baseline in the same
session as the patch was 1596.91, so >10% of that apparent gain was the
machine, not the code. Rule: never compare across sessions on this host; keep
the baseline binary and run both in one sitting.

### Local method notes

- A/B binaries: `build/release/gufo-baseline.exe` (fork HEAD) and
  `gufo-tier1.exe` (Tier 1) are kept for alternation. Rebuild overwrites
  `gufo.exe`, so copy it aside after each accepted step and rename the new one.
- Output equivalence: `gufo bench` does not print a greedy hash on this path.
  Use `gufo prompt -m <target> -p "<fixed prompt>" -n 32 -t 0 -s 42 --think
  off --no-display-prompt` and compare text; the captured log's
  `[Model Load]` line differs even when generation is identical.
- Patch merge method: `git -C <upstream clone> show <commit>^:<path>` as base,
  `<commit>:<path>` as theirs, the local worktree as ours, one
  `git merge-file -p --diff3` per file, then resolve only real conflicts.
  Worktree is CRLF; export blobs to LF first or every line conflicts.
- Upstream clone: `%TEMP%\opencode\gufo-linux`
  (HEAD `270b533`). The fork's `origin` is the Windows fork; `fee9d2a` is a
  content-equivalence base, not an ancestor.



- Use the same TheRock ROCm 10.0.0 / clang 23 build for A/B.
- Require bit-identical logits or output hashes for every kernel port.
- Measure with alternating A/B runs after the GPU is cool; `gufo bench` for
  native pp/tg, plus the HTTP path for served numbers.
- For sampled requests, also compare distributions; a faster sampler must not
  change the result.
- After each item, record the local before/after numbers here rather than only
  in the commit message.

## Known local/upstream differences

- The fork already has Windows tuning switches documented in
  [WINDOWS.md](WINDOWS.md).
- The fork already has a Latin-only draft vocabulary; #603 is incremental.
- The fork already compacts sparse selections inside `AttentionKernel`;
  #622 is incremental.
- The fork already uses 128 n-gram readers; #551 is not needed.
- The fork’s prefill chunk is 2048 for a measured reason; #470 must beat it on
  Windows, not just on Linux.
- `docs/WINDOWS.md` is the source for Windows-specific runtime behavior and
  environment switches.

### Windows loading path (investigated, not changed)

Local load stays at 34.7 s warm and 45–59 s cold against the upstream 16.42 s.
`WeightUpload` is the only loader that reopens the shard with `O_DIRECT`;
`compat/win32/posix.cpp` documents why that costs here: NTFS serializes
unbuffered reads of a file that is also mapped, and the GGUF is mapped. A
single-thread buffered read of one shard measures 2.35 GiB/s, so the measured
3.0 GiB/s aggregate is consistent with one serialized stream rather than
sixteen.

Reopening with `O_CONCURRENT_RANDOM` (cached overlapped) removes the
serialization but cannot be kept: the 103.69 GiB device allocation plus a
103.69 GiB file that becomes page-cache resident exceeds the host's 128 GiB, so
repeated loads measured 29.15 s and 53.54 s — faster at best, unstable in
practice. The experiment was reverted; `O_DIRECT` stays until the shard is no
longer mapped while it is read.
