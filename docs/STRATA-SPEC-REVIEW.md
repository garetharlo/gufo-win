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

Extended check (2026-10-26): tg512 @ 8K (126 cycles, 48.82 vs 48.79 t/s,
`output_sha256` identical) and the four `speculative-corpus` chat prompts
at 128 tokens (marginal acceptance 57-74%, the trim's regime). All eight
runs: identical SHA-256 and identical `cycles/drafted/accepted` counters —
the table changed no decision — tps within ±1.1%. The table only diverges
from the prior after ~32 observations per bin, which needs long generations
(~1k tokens); there it is the mechanism that protects against a prior that
mismatches the running quant/domain. Production value, not benchmark value.

## Evaluated and rejected

| Item | Reason |
| --- | --- |
| `--mtp-window 8192` draft-KV cap | Our draft layer already attends through the sparse indexer (`indexer_top_k = 2048` < 8192); nothing to cap. |
| hipBLASLt per-shape tuning table (gfx1151) | Strata's prefill is dense-GEMM bound; ours is MoE. `blaslt_->Gemm` serves only the unquantized projections (router, alpha/beta, indexer q/k) and the hot indexer shapes already run our own kernels (`OwnF16`/`OwnBf16`). FLOP estimate at pp4096: dense GEMMs <0.5% of prefill. Not measured with rocprof (unavailable on this Windows host). |
| Suffix-automaton drafter (~1.1k LoC) | Fallback for models without an MTP head; all our targets ship one and the fused prompt-lookup covers copy-heavy text. |
| PLE n-gram table (26.8 GiB streamed) | DeepSeek-V4-specific artifact; no equivalent tensor in our models. |
| Draft-source contention FSM | Solves prompt-lookup vs MTP as separate sources; our single fused chain encodes the same policy. |
| Probe-budget decay (5%, 0.995/round) | Our `probe_depth`/`probe_delay` mechanism is bounded differently; nothing measurable to fix. |
| Strix scheduling asymmetry (`DS4_ASYNC_T=4`, `63b8a8d3`): verify window synchronous instead of mailbox/async | Their win is dropping a cross-stream mailbox; our decode never overlaps two streams (one submission per pass, the alternate stream is picked only after `WaitFlag` returns), so there is no handoff to remove. The closest switch, `flag_waits`, is *faster* than `hipStreamSynchronize` on gfx1151; measured (UD-Q4_K_XL + shared-Q8_0 MTP, `gufo bench --n-prompt 9 --n-gen 64 --n-depth 0,16384`, 2 interleaved rounds, tg64 t/s): all-on 55.48/55.48 and 95.49/95.23, `-flag_waits` 54.65/54.76 and 93.82/93.81, `-copy_kernels` 55.22/55.21 and 94.76/95.07, `none` 54.50/54.47 and 93.13/93.48. Every Windows switch helps; the package is worth +1.4% at depth 0 and +2.5% at 16K, the opposite sign of Strata's result. Not re-tested on Linux (no gfx1151 Linux host here), where the same switches are off by default. |

## Strata findings worth remembering

- Overlapping drafter and verify on separate streams (`--spec-split`)
  *hurts* on their Strix/A770 measurements (9.9-10.3 vs 11.7-12.6 tok/s).
  Do not propose stream overlap for the draft head.
- Spec width beyond 4 was worse and `spec-min-p ~0.5` optimal in their
  sweeps; both match our controller's converged behavior.
- Their conclusion for prefill/decode headroom: per-layer handoff latency
  and CPU-pool/GPU overlap, not kernel rewrites.
