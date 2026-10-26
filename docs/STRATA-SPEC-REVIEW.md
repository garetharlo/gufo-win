# Strata speculative-decoding review (2026-10-26)

Review of `github.com/niko1221/strata` @ `4b9e05a` (2026-10-25) for
speculative techniques portable to Gufo's qwen3.8-flash-next decode path.
Strata's speculative stack: `src/spec/` (controller, draft policy, suffix
drafter), `src/ngram/` (PLE table reader), docs in `docs/A770_SPEED_SPEC.md`
and `docs/STRIX_HALO.md`.

## Ported

| Item | Commit | Notes |
| --- | --- | --- |
| Online accept-probability table | `df097b30` | `MtpAcceptTable`: Beta hit/miss counts per (depth-class, top-prob bin), seeded with the 2026-09-25 prior at weight 32, refined from every verified cycle. Replaces the frozen `kMtpAcceptByTop` lookup feeding the survival trim. Output-invariant (draft-side only; `output_sha256` identical in A/B). |
| Stall clamp on cycle-cost samples | `df097b30` | `MtpBatchController::Observe` clamps any cycle-ms sample >4x the bin's running estimate before the 0.25-EMA, so one preemption/page-fault spike cannot close a width for an occupancy/context bin. Strata's measured constant. |

A/B (`tools/windows/bench-flash-next-context.ps1`, 1 rep, deterministic
seeds; report: `artifacts/bench-context/spec-ab-report.html`, commit
`6e8399df`): tg128 MTP 50.17 -> 51.13 t/s @ 8K, 93.88 -> 93.54 @ 32K; both
within single-run noise (untouched AR control moved +4.2%). The runs are
too short to fill the table (~32 observations per bin); throughput-neutral,
output-identical.

## Evaluated and rejected

| Item | Reason |
| --- | --- |
| `--mtp-window 8192` draft-KV cap | Our draft layer already attends through the sparse indexer (`indexer_top_k = 2048` < 8192); nothing to cap. |
| hipBLASLt per-shape tuning table (gfx1151) | Strata's prefill is dense-GEMM bound; ours is MoE. `blaslt_->Gemm` serves only the unquantized projections (router, alpha/beta, indexer q/k) and the hot indexer shapes already run our own kernels (`OwnF16`/`OwnBf16`). FLOP estimate at pp4096: dense GEMMs <0.5% of prefill. Not measured with rocprof (unavailable on this Windows host). |
| Suffix-automaton drafter (~1.1k LoC) | Fallback for models without an MTP head; all our targets ship one and the fused prompt-lookup covers copy-heavy text. |
| PLE n-gram table (26.8 GiB streamed) | DeepSeek-V4-specific artifact; no equivalent tensor in our models. |
| Draft-source contention FSM | Solves prompt-lookup vs MTP as separate sources; our single fused chain encodes the same policy. |
| Probe-budget decay (5%, 0.995/round) | Our `probe_depth`/`probe_delay` mechanism is bounded differently; nothing measurable to fix. |

## Strata findings worth remembering

- Overlapping drafter and verify on separate streams (`--spec-split`)
  *hurts* on their Strix/A770 measurements (9.9-10.3 vs 11.7-12.6 tok/s).
  Do not propose stream overlap for the draft head.
- Spec width beyond 4 was worse and `spec-min-p ~0.5` optimal in their
  sweeps; both match our controller's converged behavior.
- Their conclusion for prefill/decode headroom: per-layer handoff latency
  and CPU-pool/GPU overlap, not kernel rewrites.
