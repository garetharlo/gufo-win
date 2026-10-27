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
- The shared-expert lane turns itself on with the attention classes it can handle.
- The power and memory guard is on by default (`GUFO_NPU_GUARD` accepts
  `on`, `off` or `log`). It holds the NPU for 600 s when three consecutive
  chunks average above 82 W, and it reads whole-chip power, so a host whose
  GPU prefill alone exceeds 82 W holds the NPU after three chunks. Measured
  on a 32 GB BOSGAME host: NPU 1563 tok/s against 1608 for the GPU alone at
  `pp8192`, 1528 with the guard off, so the holds reduce the loss; the
  offload only pays off when the NPU beats the GPU for its own slices.

## NPU power mode

Gufo reads the power mode and writes it to the log once. To see it, run `C:\Windows\System32\AMD\xrt-smi.exe examine --report platform` and read the line `Power Mode`. The measured gain was the same in Default and Performance mode. To change the mode, run `xrt-smi configure --pmode performance` (or `default`) in an administrator shell. The mode resets when the driver reloads.

## Licenses

The NPU kernel and design come from mikealanni/gufo (`tools/npu`, MIT) and are derived from AMD mlir-aie programming examples (Apache-2.0 WITH LLVM-exception, Copyright (C) 2026 Advanced Micro Devices, Inc.). The license texts are in the `licenses` folder.
