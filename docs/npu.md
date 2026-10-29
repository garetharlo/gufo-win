# NPU prefill

Gufo can send part of the prompt processing (prefill) to the Ryzen AI NPU while the GPU does the rest. NPU prefill is off by default. To turn it on, set the environment variable `GUFO_NPU_PREFILL=1` for the server process. It needs the NPU, its driver and the kernel files. If the NPU is on and cannot start, Gufo uses the GPU path.

## What to expect

- The NPU takes only full blocks of 4,096 prompt tokens. Prompts shorter than 4,096 tokens do not use it.
- Generation speed (decode) does not change. The NPU is used for prefill only.
- Agent work changes very little. In an agent session, each turn adds only a few hundred new tokens to a cached conversation, so most turns are below 4,096 new tokens. The NPU helps the first turn of a session and large pasted files or documents.
- The NPU path uses 8-bit block floating point for its part of the work. Output is not byte-exact.
- The NPU keeps about 1.2 GB of weights in memory.

## Requirements

- The AMD NPU driver and runtime (XRT). Gufo does not install them.
- Tested with Windows 11 build 26200, NPU driver 32.0.20102.3930 (2026-05-06) and XRT 32.00.20102.3931. Gufo uses the NPU only with AMD XRT runtime 32.0.20102.3931. With any other runtime, Gufo writes a warning, reports the NPU as faulted, and uses the GPU. The output does not change.
- The kernel files `final_4096x2560x1024_128x64x128.xclbin` and `insts_4096x2560x1024_128x64x128.txt`. `GUFO_NPU_DIR` names the folder that holds them; the sources `mm_bfp_mixed.cc` and `n32_core_placed.py` are in the same folder.

## Turn it on or off

- Off: `GUFO_NPU_PREFILL` unset or not `1` (the default).
- On: set `GUFO_NPU_PREFILL=1` for the server process, and set
  `GUFO_NPU_DIR` to the folder that holds the kernel files when it is not
  `bin\..\tools\npu` (a source build keeps them in `tools\npu`, so set it).
  Without usable files Gufo writes one warning and uses the GPU.
- The attention q/gate lane is off by default. Turn it on with the environment variable `GUFO_NPU_GATE_SLICES` (1 to 12).
- The shared-expert lane is off by default; `GUFO_NPU_SHEXP=1` turns it on. Its
  gate/up matrices are at most 1024 rows, which the GPU finishes faster than one
  NPU dispatch takes, and the GPU stream waits for every one of them: 7 % slower
  prefill when it is on.
- The power and memory guard reports by default (`GUFO_NPU_GUARD` accepts `on`,
  `off` or `log`). `on` holds the NPU for 600 s when three consecutive chunks
  average above 82 W. It reads whole-chip power, so on a desk host the GPU
  prefill alone trips it: 1659 tok/s held against 1690 un-held at `pp8192`.
- Measured on a 32 GB BOSGAME host, `pp8192` of `UD-Q4_K_XL`: 1690 tok/s with
  the NPU against 1656 for the GPU alone, decode unchanged. The NPU runs the
  projections it takes at about 2 % of the GPU's rate, so the gain is a few
  percent of prompt processing, not a substitute for the GPU.
- One chunk is the engine's prefill chunk (8192 tokens by default) and a lane
  takes every whole 4096-token group in it.

## Where the batch time goes

Every prefill chunk that used the NPU logs one `npu request:` line. The
`batch[...]` fields are phase sums inside the XRT batch (all lanes, request
total): `slices` = GEMM dispatches of the 4096-token groups, `weight_ms` =
weight (B) uploads or resends (~0 once resident), `fill_ms` = activation (A)
upload, `start_ms` = `run.start()` calls, `wait_ms` = `run.wait()` (device run
plus queue), `d2h_ms` = output (C) sync back, `consume_ms` = host callback on
the result.

Reading: `weight+fill+start+consume` is host work, `wait+d2h` is device work.
If host far exceeds device, the fix is fewer or fused dispatches, not a faster
kernel; if `wait` dominates and is near the int8 tile peak, only the kernel
itself limits. Four runs per lane are already in flight, so a healthy lane
shows `wait_ms` smaller than the wall prefill time by design.

### Probe log

Record each run here so the same question is never measured twice.

| date | host | build | command | slices | weight ms | fill ms | start ms | wait ms | d2h ms | consume ms | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-28 | Ryzen AI MAX+ 395, 32 GB, Balanced power | release | `gufo bench --n-prompt 16384 --n-gen 32 --repetitions 1 --temperature 0` (UD-Q4_K_XL) | 432 | 0.2 | 72.3 | 14.2 | 343.8 | 43.9 | 136.1 | GPU-bound, see below |

### What the 2026-10-28 probe ruled out

One 8192-token chunk (two 4096 groups, 432 slices) took 5027 ms wall; the
NPU batch phases above sum to 611 ms, so the NPU is busy about 7 % of the
chunk and the GPU owns the rest. `hostfunc_lat_ms=1674` is the GPU linear-Z
callback latency, i.e. the GPU work the NPU replaced, and it runs alongside
the MoE and attention streams instead of extending them - which is why
removing it changes nothing.

- Measured: pp16384 1648.14 tok/s with the NPU against 1646.28 without
  (+0.11 %, noise), tg32 unchanged.
- Ping-pong staging (overlapping `fill` with `wait`) can recover at most the
  226 ms of host/DMA per chunk, about 4.5 % of prefill, and only if the
  device is the one waiting. Not worth a rewrite of the staging path yet.
- Fused AIE kernels speed up the NPU's 344 ms share, which is off the
  critical path, so they cannot move pp either.
- The remaining lever is to give the NPU work the GPU is actually blocked on
  (the MoE experts) rather than making the projection path faster.

## NPU power mode

Gufo reads the power mode and writes it to the log once. To see it, run `C:\Windows\System32\AMD\xrt-smi.exe examine --report platform` and read the line `Power Mode`. The measured gain was the same in Default and Performance mode. To change the mode, run `xrt-smi configure --pmode performance` (or `default`) in an administrator shell. The mode resets when the driver reloads.

## Licenses

The NPU kernel and design come from mikealanni/gufo (`tools/npu`, MIT) and are derived from AMD mlir-aie programming examples (Apache-2.0 WITH LLVM-exception, Copyright (C) 2026 Advanced Micro Devices, Inc.). The license texts are in the `licenses` folder.
