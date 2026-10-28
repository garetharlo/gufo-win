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

pp8192, UD-Q4_K_XL, 32 GB BOSGAME host, guard on where noted:

| build                        | tok/s   |
| ---------------------------- | ------- |
| GPU only                     | 1608.09 |
| NPU + guard                  | 1563.61 |
| NPU, guard off               | 1527.68 |
| upstream 0.1.0 binary        | 1520.89 |

The guard calibration predates `1018b08f` (`kPrefillChunkTokens` 4096 → 8192,
+2.2 % pp8192 GPU only). Decisions are unaffected; only the magnitudes are
stale.

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
