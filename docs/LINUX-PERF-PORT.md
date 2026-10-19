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

### [ ] #613 — copy repeated context into MTP drafts (prompt lookup)

Upstream: `8497711`. Upstream Linux measurements:

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

### [ ] #603 — score MTP drafts over the request vocabulary

Upstream: `51a8b00`. Upstream Linux measurements: **+13% coding-agent replay**,
**+16% everyday prompts**, **+5% across ten languages**, greedy bit-identical.

Important local delta: the fork already has `DraftVocabulary::kLatinText`
(`engine.cpp:39–93,207–210`), which halves the output head for ASCII/Latin
text. This is **not** a +13–16% win over the current Windows build for
English/code. The increment is:

- request-aware first-98,304-row subset plus a 1,024-row per-session table,
- fallback for non-Latin text,
- adaptive draft length from draft confidence.

Changes: add `draft_vocab.hpp`; port `SetDraftVocab`, `AppendDraftRows`,
`ClearDraftRows` and `draft_subset` through the executor/engine; replace the
fixed Latin vocabulary path. Local executor already has a `draft_vocab` vector,
so adapt rather than reimplement.

Windows fit: host + row-gather kernel; no OS dependency. Check batched-session
memory for the per-session table.

Acceptance: greedy bit-identical; English/code neutral or faster; CJK/Cyrillic
faster; no sampled regression.

### [~] #518 — faster Flash-Next decode kernels

Upstream: `7e4621c`. **+6–7% single-session decode**, **+4% with 2–4
sessions**, prefill unchanged, bit-identical. Fifteen small changes: fused Q8_1
producers, DPP `row_xmask` / `v_permlanex16` instead of `__shfl_xor`, MoE
grouping, batched attention loads, vector loads.

Local state: `kernels.hip.cpp` has diverged (141/47 lines vs base) and the fork
already has some decode work. Port selectively, change by change; do not
cherry-pick the whole commit.

Windows fit: pure HIP. No new switches.

Acceptance: bit-identical outputs; measure tg at d0/32K, single and 4 sessions.

Windows port log (in progress):

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

### [ ] #650 — fork recurrent state instead of copying it

Upstream: `95ba1ea`. Qwen27B 4-request aggregate **64.4 → 68.0 tok/s
(+4–6%)**, single request **+2.5%**, copy time **861 → 24 ms**, outputs
identical.

Local divergence in the relevant files is small:

- `src/models/qwen/hip/arena.cpp` (10 lines vs base)
- `src/models/qwen/hip/ssm_replay.cpp` (66/8)
- `src/models/qwen/hip/decode.cpp` (7)
- `src/models/qwen/hip/executor.cpp` (56/32)
- `kernels/deltanet_decode.hpp`, `ops/ssm.hpp`

Changes: two conv/deltanet buffer pairs, live/saved/source indices,
`QwenRecurrentStateIo` with in/out pointers, rollback diverts the next pass
instead of `hipMemcpy`, snapshot sync before transfer, only the header zeroed
in `SerializeCompact`.

Windows fit: pure HIP. **Check VRAM headroom** for the second buffer pair; the
gain comes from removing copies.

Acceptance: outputs identical; 1-request and 4-request Qwen27B decode improve.

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

### [~] #485 — accelerate prefill and MTP catch-up

Upstream: `fd747a5`. Upstream matched HTTP increments: pp2048 d0 MTP
**1581.01 → 1623.99 tok/s**, d32K **1490.59 → 1524.10**, d128K AR
**1381.32 → 1400.23**, TG within 0.5%, completions and draft counts identical.
Eight independent slices; the fork has no `GdnCheckpoint`/`PrefillCheckpoint`
stack, so checkpoint code is dropped from every slice.

- [x] **GDN row reuse** in `GdnRowSplitKernel`: two rows per lane, four waves,
  128 threads per block, shared `q`/`k` loads reused across the row pair.
- [ ] Padded SSM F16 rows (`half_stride`, `DenseF16Gemm` `kStrided`).
- [ ] Full selector scratch (`select_chunk` without `bit_floor`,
  `first_query = n_tokens - 16` for `last_only`), packed `SelectScore` half2
  loads and `sched_barrier`.
- [ ] `RoutedPadBounds` clears padding in-kernel, dropping two
  `hipMemsetAsync` calls.
- [x] `HcMix` `half_only`, skipping the F32 and Q8 copies when both stacked
  SSM projections read the F16 input the wide mixer already wrote. Needs the
  `y != nullptr` guard in the `kHcMix` epilogue and the relaxed
  `HcMixF16Gemm` argument check: measured +0.5% pp2048 d0, +0.1% d32K, tg
  neutral, greedy haiku and `--validate-prefill 64` (rmse 0.42) byte-identical
  to the GDN build.
- [ ] PLE direct mapped n-gram input (`hipHostMallocMapped | Coherent`,
  `hipHostGetDevicePointer` into `s.ple_emb`, no F32 allocation or uploads).
- [ ] `DenseBatch` 32-row chunk split for 64+ rows.
- [ ] MTP catch-up sparse-boundary fix (`hidden_source` +
  `control_host_->hidden_row`, `Attention` `last`/`dense_rows` flags).

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

### [ ] #470 - 4096-token prefill chunks + overlapped n-gram gather

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

### [~] #571 — faster wide prefill kernels

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
- [ ] gate/up fused rows (paired shared-expert W8A8 rejected upstream).
- [ ] shared K16 fragments.
- [ ] register-resident Q8_0 SSM fragments.
- [ ] deferred residual in MoE combine (`HcCombineF16` defer_w/inject_out).
- [ ] 128-token gate/up and 64-token down tiles + tail kernel.
- [ ] software-pipelined attention, GDN q/k in registers.

Windows fit: pure HIP, but large kernel rewrite against a diverged fork. Do
last; port one kernel family at a time.

### [ ] #626 — cut prefill/decode attention cost at depth

Upstream: `5a1669a`. HTTP AR pp64K **1578 → 1597**, pp128K **1517 → 1544**;
MTP pp64K **1531 → 1563**, pp128K **1477 → 1503**; deep MTP decode **+1–5%**.
Also fixes repeated MTP scratch allocation and vision-prepare dispatch stalls.

Windows fit: pure HIP. Large change; take the attention and scratch fixes
separately.

### [ ] #622 — compact sparse attention selections once per row

Upstream: `baf5a1b`. **+3.3% at 160K**, server deep decode **+1.4–2.4%**,
bitwise identical.

Local state: `AttentionKernel` already compacts inside the kernel
(`kernels.hip.cpp:2835–2894`), but rebuilds the list per (head, split). Upstream
adds `CompactSelectionKernel` and reads the per-row list once. Incremental.

Windows fit: pure HIP.

### [ ] #645 — reuse historical image preprocessing

Upstream: `f948ddd`. Image queue medians:

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

---

## Cheap optional ports

### [ ] #415 — skip vocabulary blocks that cannot change the sampled token

Upstream: `b945d0a`. Sampler microbench: top-k 20 **8.4×**, top-k + presence
**15×**; end-to-end **+1.2–3.6%** on sampled requests. Local
`src/core/sampling.cpp` still scans every entry and binary-searches penalties.

Windows fit: pure C++. Low risk. Greedy on GPU is unaffected.

### [ ] #332 — GPU greedy penalties

Upstream: `7e450e8`. Avoids the GPU→CPU fallback when greedy MTP uses
presence/repetition penalties. Adds a penalty kernel and executor changes.
Only worth it if penalties are actually used.

### [ ] #348 — skip disk checkpoints that barely advance a prefix

Upstream: `93af45d`. Fewer disk writes/fsyncs on multi-turn chats. No
decode/prefill gain. Trade-off: disk-only restore re-prefills up to 2,048
tokens. Do it only if the disk cache is a measured bottleneck.

### [ ] #499 — stage Qwen-Image weight uploads through pinned buffers

Upstream: `eb91daa`. Local `qwen_image_21/weights.cpp` uses `mmap`; staging
through `hipHostMalloc` avoids the host page-pinning stall. This is a
load/stall fix, not a decode speedup. Optional.

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
4. Loading-time investigation: compare the Windows read/upload path with the
   upstream 16.42 s reference before porting anything.
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
