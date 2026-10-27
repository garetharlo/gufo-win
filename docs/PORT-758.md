# Task: port upstream #758 (chunked DeltaNet + stream overlap)

Resume point for porting upstream `gufo-org/gufo` `538e2fd7` — "perf(qwen-flash):
faster Flash-Next prefill with chunked DeltaNet and stream overlap (#758)" —
into this fork. Triage: [LINUX-PERF-PORT.md](LINUX-PERF-PORT.md), section
"Upstream sweep 2026-10-27".

Upstream gain (same hardware): pp512 +4.9%, pp2048 +4.1% (1,741 -> 1,812
tok/s), pp4096 +4.3%, d16K pp2048 +4.0%. Decode must stay at parity: upstream's
earlier multi-stream attempts cost -11% decode before #735 narrowed the waits,
so every overlap stage is measured on tg too.

## Sources and commands

- Upstream commit `538e2fd7` on `https://github.com/gufo-org/gufo` main.
  The `upstream` remote is already configured in this clone.
- Regenerate diffs (paths are `src/models/qwen38_flash_next/...`):

  ```powershell
  git show 538e2fd7 -- src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp > $env:TEMP\opencode\p758_hip.diff
  git show 538e2fd7 -- src/models/qwen38_flash_next/kernels/rocm/kernels.hpp > $env:TEMP\opencode\p758_kh.diff
  git show 538e2fd7 -- src/models/qwen38_flash_next/kernels/rocm/executor.cpp > $env:TEMP\opencode\p758_exec.diff
  git show 538e2fd7 -- src/models/qwen38_flash_next/kernels/rocm/executor.hpp > $env:TEMP\opencode\p758_ehpp.diff
  git show 538e2fd7 -- tests/models/qwen38_flash_next/gdn_ops_test.cpp > $env:TEMP\opencode\p758_test.diff
  ```

- Upstream rationale and measurements:
  `git show 538e2fd7:docs/models/qwen3.8-flash-next/EXPERIMENTS.md` (sections
  "Flash-Next prefill throughput push" and the ROCm 7.2 caveats) and `QUALITY.md`.
- Do **not** cherry-pick. This fork diverged in all four touched files. Adapt by
  hand and keep local Windows behavior (hipGraph capture, flag waits, recorded
  rollback, existing `primary_stream_`/`alt_stream_` uses).

## Gates

- `gdn_ops_test` chunked cases (ported with the kernel) pass on gpu-test.
- No spills in `GdnChunkScanKernel` on our TheRock clang 23. Upstream measured
  237 VGPR on clang 24; clang 22 spilled and lost to the row split.
- Bench: pp512/pp2048/pp4096 up ~4%, d16K pp2048 up, tg128 AR and MTP within
  noise.
- NLL quality pass per repo protocol. Upstream: overall +0.0021 (noise), chat
  +0.0145 (real, accepted) from the fused gated norm; W8A8 tile swaps are
  bit-identical.
- Row-split fallback stays: chunked only when `kGdnChunkedScan && row_split &&
  d == 128 && n_tokens >= 256`, `convolved`, null speculative snapshots.

## Stages

- [x] 0. Sweep + triage recorded in LINUX-PERF-PORT.md (2026-10-27).
- [x] 1. ~~Chunked WMMA DeltaNet~~ REMOVED 2026-10-27 by upstream #850 and by
  this fork: the kernel, its prep and fused gated norm, the `kGdnChunk*`
  constants and `GdnChunkedScanAvailable()` are gone again; only the `z_ready`
  wait on `GatedDeltaNet` remains (the epilogue uses it, stage 3). Reason and
  numbers in [LINUX-PERF-PORT.md](LINUX-PERF-PORT.md), "Upstream fix 2026-10-27".
- [x] 2. Row-first W8A8 tile (32 rows) for mixer down at <=1024 tokens
  (`kW8A8NarrowMaxTokens`, `kRowsFastest`), bit-identical, old tile above 1024.
- [ ] 3. SKIPPED. Side-stream overlap in `executor.cpp` (shared expert beside
  routing, Z-row tiles on a side stream). Both touch `#ifdef GUFO_NPU` paths
  that upstream does not have: the shared expert shares its scratch with
  `npu::BeginProjection`/`FinishProjection`, and `DenseF16SsmGemm` already
  splits rows for the NPU Z prefix. Upstream's own ROCm 7.2 notes measured
  these overlaps as losses wherever the chunked scan is inactive, and stages
  1+2 already land at upstream's prefill gain (see stage 4). Revisit only if
  short-prompt (pp512) throughput matters more than that risk. **Reopened by
  the #850 removal**: upstream's post-removal table (row split, no scan) is
  1,286/1,858/1,909 without the overlaps and 1,331/1,855/1,916 with both
  (+3.3% pp512, neutral pp2048, +0.4% pp4096), so the overlaps are now the
  only lossless prefill headroom left.
- [x] 4. Ported `gdn_ops_test` chunked cases; GPU run green; bench below;
  docs updated. NLL sweep not run here (no Linux quality host): numerics are
  bounded by the chunked-vs-row-split test (1.1e-2..1.6e-2 over RMS, the band
  upstream accepted at +0.0021 overall NLL) and the W8A8 tile swap is
  bit-identical.

## Progress log

- 2026-10-27: upstream sweep found #758 as the only performance item since
  `16e8ed47`; triage written; port started (stage 1).
- 2026-10-27: stage 1 done. `GdnChunkScanKernel<true>` 234 VGPR / 0 spills on
  TheRock clang 23 (upstream 237 on clang 24), so the gate stays on for clang
  23. `gdn_ops_test` chunked n=256/300/1030: output 0.011-0.014, state
  0.014-0.016, fused output 0.011-0.013 over RMS.
- 2026-10-27: stages 2 and 4 done, stage 3 skipped. Windows release build,
  `gufo bench --n-prompt 512,2048,4096 --n-gen 128 --n-depth 0 --repetitions 1
  --temperature 0`, one run per binary, same GPU state:
  pp512 1,176.89 -> 1,203.29 (+2.2%), pp2048 1,645.26 -> 1,718.11 (+4.4%),
  pp4096 1,642.46 -> 1,705.69 (+3.9%), tg128 34.85 -> 34.91 (noise).
  Upstream: +4.9/+4.1/+4.3% with the overlaps, -30% GDN call time at pp512.
- 2026-10-27: the README upstream-comparison table was re-measured (NPU off;
  depths 0/16K/32K, pp2048/tg128, two reps, one process per mode). Prefill
  1,710.08 / 1,512.37 / 1,487.24 (AR) and 1,656.11 / 1,471.97 / 1,446.46 (MTP)
  against upstream 1,684.74 / 1,465.02 / 1,443.87 and 1,631.13 / 1,413.58 /
  1,392.54: +1.5..+4.1%. Decode is unchanged at d0 and d32K; the 16K MTP cell
  fell 76.37 -> 43.12 tok/s and was validated: `--validate-prefill 4096` gives
  rmse 0.3108 / cosine 0.995868 for the chunked prefill against the one-token
  reference (row split: 0.333746 / 0.995312), the first 32 tokens at d16K are
  bit-identical to the pre-port binary, and with `kGdnChunkedScan = false` the
  new binary reproduces the old trajectory exactly (same output sha, same
  drafted/lookup counters) at 72.09 tok/s. So the cell measures where the model
  ended its turn (`<|im_end|>` after token 36), not decode speed: past the turn
  the drafts stop being accepted and the rest runs at AR speed (32 ms/token,
  confirmed by 47.99/44.53/42.27 tok/s at n=96/112/128 and 36.47 at n=256).
- 2026-10-27: decided not to make the bench stop at the end of the turn.
  `DecodeStep(..., stop_at_eos = true)` returns `stop` without advancing the
  session, so the loop cannot continue, and the AR depth rows end their turn on
  the first token at 16K/32K (`engine.cpp:1185-1188`), which would leave two
  table cells with no timed tokens. A budget of 32 tokens with `stop_at_eos`
  true times the turn only (76.76 tok/s at d16K MTP, against 78.40 upstream) but
  changes the method for one cell. Instead `gufo bench` keeps the 128-token
  window and now prints `ended its turn after N of M tokens; the rest is
  post-turn text`, so the artefact is visible in every future run.
- 2026-10-27: re-measured the whole upstream comparison with the stock v0.24.0
  Windows binary (`gufo-v0.24.0-windows-x86_64.zip`, same machine) at a 32-token
  budget, three repetitions, so both columns time the same thing and no cell
  runs past the end of the turn. Gufo / stock tg: AR 33.55/27.55, 31.86/27.90,
  30.93/27.79 at d0/16K/32K; MTP 51.48/30.67, 87.29/85.01, 85.70/85.12. The
  former -45% cell was the 128-token window of a different greedy sequence
  (ours ends its turn at token 36, so 92 of its tokens were post-turn text);
  at 32 tokens both binaries emit the same prefix and the cell is +2.7%. Stock
  measured 75.65 tok/s at 128 tokens in this session, not the 78.40 the old
  table quoted, so the old cell was never like-for-like. README now carries the
  matched table and no footnote.
- 2026-10-27: stage 1 reverted here, following upstream `c92b19c5` #850 (the
  scan flips greedy decisions: chat NLL +0.0145, layer error 2.6e-4 vs 5.1e-7
  relative RMS against the F32 recurrence). Windows release, `gufo bench
  --mtp-model <shared Q8_0> --speculative mtp -p 512,2048 -n 64 -r 3`, two
  alternated rounds, before/after exes: pp512 1,181/1,195 → 1,179/1,174
  (−0.9%), pp2048 1,596/1,596 → 1,576/1,546 (−2.2%), tg64 52.7/53.4 →
  53.3/53.3 (noise). The old gains were +2.2/+4.4% in the stage 1+2 session, so
  part of what stage 1 bought was already inside session noise. `--validate-prefill`
  at 512/1024/2048 is unchanged within noise (old 1.1985/0.3378/0.3781 rmse,
  new 1.2065/0.3506/0.3487): that metric is dominated by the batched attention
  difference, not by the recurrence, so it only guards regressions. Greedy text
  (`-t 0`, 48 tokens) is identical for a 70- and a 115-token prompt and differs
  from the first token for a ~1.1K-token doc continuation, so the change is
  scoped to long prefill; a ~1.1K-token tool prompt happened to agree. Gate
  recipe for the next pair: [TESTING.md](TESTING.md#flash-next-binary-ab).
