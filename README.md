# Gufo Windows: the Strix Halo inference engine

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

Gufo is a vertical local inference engine specifically built and optimized for
the AMD Strix Halo hardware: Ryzen AI MAX+ 395 systems with Radeon 8060S
(`gfx1151`), up to 128 GiB of unified memory. This fork builds and runs it
natively on Windows 11 x64 against AMD's TheRock ROCm — no WSL, no
containers. The Linux original lives at
[upstream gufo](https://github.com/pixmaate/gufo), and the port details are in
[docs/WINDOWS.md](docs/WINDOWS.md).

Contributions are welcome!

## Models and benchmarks

All model documentation lives under [docs/models](docs/models/README.md):

| Model | Inference modes | Hugging Face weights | Benchmarks | Quality |
| --- | --- | --- | --- | --- |
| [Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) | Q4/Q8, images, AR, DFlash2 | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf) / [Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q8_K_XL.gguf) · [DFlash2 Q4_K_M](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/blob/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) | Q4: **656.33 tok/s pp**; up to **70.56 tok/s tg** single user and **123.00 aggregated tok/s** on 8 concurrent requests with DFlash2 · [Benchmarks](docs/models/qwen3.8-27b/BENCHMARKS.md) | [Quality](docs/models/qwen3.8-27b/QUALITY.md) |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md) | Q4, images, AR, MTP | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL) · [MTP Q8_0](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf) | **1,628.52 tok/s pp**; up to **59.41 tok/s tg** single user and **157.22 aggregated tok/s** on 8 concurrent requests with MTP; direct 8K–256K sweep: **1,600.90 tok/s pp**, up to **75.38 tok/s MTP tg** (repeated-text upper bound, 10.6–19.6% faster than bubo 0.1.0) · [Benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md) · [Context sweep](docs/models/qwen3.8-flash-next/bench-context.html) | [Quality](docs/models/qwen3.8-flash-next/QUALITY.md) |
| [Qwen3.6-35B-A3B](docs/models/qwen3.6-35b-a3b/README.md) | `qwen35moe` GGUFs, images, AR | See [its guide](docs/models/qwen3.6-35b-a3b/README.md) | In progress · [Benchmarks](docs/models/README.md) | In progress |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash/README.md) | AR, DSpark | [antirez Flash 0731 IQ2XXS](https://huggingface.co/antirez/deepseek-v4-gguf/blob/1cd7b564460821938add0475a60b942c409295e0/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf) · [DSpark](https://huggingface.co/antirez/deepseek-v4-gguf/blob/e7f04037032990db0346398d249baf9fb9df1ccc/DeepSeek-V4-Flash-DSpark-support-0731.gguf) | **484.62 tok/s pp**; up to **26.62 tok/s tg** single user and **54.74 aggregated tok/s** on 8 concurrent requests with DSpark · [Benchmarks](docs/models/deepseek-v4-flash/BENCHMARKS.md) | [Quality](docs/models/deepseek-v4-flash/QUALITY.md) |
| [Qwen3-ASR 1.7B](docs/models/qwen3-asr/README.md) | Speech recognition | [BF16](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5) | **15.27× realtime** · [Benchmarks](docs/models/qwen3-asr/BENCHMARKS.md) | [Quality](docs/models/qwen3-asr/QUALITY.md) |
| [Qwen3-TTS 1.7B](docs/models/qwen3-tts/README.md) | Speech synthesis and voice cloning | BF16 [CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) / [VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) / [Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Up to **2.54× realtime**; **201 ms** to first audio (CustomVoice) · [Benchmarks](docs/models/qwen3-tts/BENCHMARKS.md) | [Quality](docs/models/qwen3-tts/QUALITY.md) |
| [Qwen-Image-2.1](docs/models/qwen-image-2.1/README.md) | BF16 image generation and editing | [Complete pipeline](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/b3179ad355be050328e483a9dfdd9e60cd62adfa) | In progress · [Benchmarks](docs/models/qwen-image-2.1/BENCHMARKS.md) | [Quality](docs/models/qwen-image-2.1/QUALITY.md) |
| [MiniMax H3](docs/models/minimax-h3/README.md) | BF16 text to video/audio | [FL2VA pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA) | In progress · [Benchmarks](docs/models/minimax-h3/BENCHMARKS.md) | [Quality](docs/models/minimax-h3/QUALITY.md) |

**Running on Windows today:** `start.cmd` serves Qwen3.8-Flash-Next (adaptive
MTP, images), Qwen3.8-27B (DFlash2) and Qwen3.6-35B-A3B. DeepSeek V4 Flash and
the audio/video/image pipelines are documented for the engine but not yet
validated on Windows.

Peak measured workloads; text pp is autoregressive (AR), while tg uses the
named speculative mode. Peaks include repetitive output; aggregate tg sums
individual request decode rates. Qwen27B's single-user peak uses the short-prompt
C1 workload. Audio excludes loading. The Flash-Next context sweep uses direct
`gufo bench` runs, so served rates differ (`start.cmd` enables survival
stopping, the Latin draft vocabulary and prompt lookup); each model guide lists
the required files and complete benchmark settings.

## Philosophy

We would like this to be the one-stop shop for Strix Halo Local AI
enthusiasts: batteries included for text, audio, image, and video models.

- Build and optimize specifically for the Strix Halo 128 GiB hardware. Smaller memory configurations should still work and preserve the speed benefits for models that can fit on memory.
- Support only the best available models for their size that can run on this hardware: less code to maintain, more focused optimization and testing work.
- Preserve quality when optimizing. Each model's quality report records independent numerical checks, execution consistency and unresolved gaps. Don't reuse kernels across different models to limit blast radius of a code change.
- Treat concurrent requests, cancellation and conversation caching as first-class workloads.
- Keep production dependencies small and development tools separate.
- Contributions are welcome — we need the Strix Halo community to keep improving gufo!

## Quickstart

1. Reserve dedicated GPU memory for the model you want (AMD Software >
   Performance > Tuning > Variable Graphics Memory): **96 GB** runs
   Qwen3.8-Flash-Next at the full 256K context and decode speed.
2. Download a prebuilt `gufo-windows-<date>.zip` from
   [Releases](https://github.com/jaacMusic/gufo-win/releases) and unpack it.
   `gufo.exe` carries its ROCm, vcpkg and VC++ runtimes; it needs only a Strix
   Halo machine, a current AMD driver and the models. Its `README.txt` lists
   the downloads.
3. Download the qualified Flash-Next files with the Hugging Face CLI
   (`pip install -U huggingface_hub`):

```powershell
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf"
run start.cmd
```

`start.cmd` lists the servable models it finds with their draft and vision
sidecars, number keys toggle features (speculative decoding, prompt lookup,
survival stopping, the Latin draft vocabulary, images, thinking, context,
network access) and Enter starts the server on
`http://127.0.0.1:8080/v1`; it serves OpenAI-compatible Chat Completions
(text, images, tools, streaming) and Responses (text, streaming). The scriptable
form is `tools\windows\run-flash-next.ps1`.

Then, from another terminal, ask it something:

```powershell
curl http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\": \"Qwen3.8-Flash-Next-UD-Q4_K_XL\", \"messages\": [{\"role\": \"user\", \"content\": \"Say something\"}]}'
```

For Open WebUI, VS Code and OpenAI SDK clients, set the API base URL to
`http://127.0.0.1:8080/v1`. See the [API contract](docs/SERVER.md).

The text server uses the model's native context by default and generates until
EOS or the context is full. `--context N` sets context capacity per session;
`--max-tokens N` sets a default response limit that clients can override.
Reasoning tokens count toward that response limit.

## Build from source

Windows 11 x64 on AMD Strix Halo (`gfx1151`), with test statuses, memory notes
and port internals in [docs/WINDOWS.md](docs/WINDOWS.md).

| Piece | Default location | Notes |
| --- | --- | --- |
| TheRock ROCm 10.0.0 (Windows, gfx1151) | `C:\TheRock\build` | [`therock-dist-windows-gfx1151-10.0.0.tar.gz`](https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-10.0.0.tar.gz); extract so that `C:\TheRock\build\bin` exists. Its clang compiles C, C++ and HIP |
| vcpkg | `C:\vcpkg` | `vcpkg install icu curl openssl libpng libjpeg-turbo --triplet x64-windows` |
| Visual Studio Build Tools | any | "Desktop development with C++": MSVC STL + Windows SDK only |
| CMake 3.21+ and Ninja | `PATH` | |

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1   # what is missing, what fits, how to fix it
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1   # release -> build\release\gufo.exe
```

`build.ps1` imports the MSVC environment, configures with TheRock clang and
the vcpkg toolchain, builds, and copies the ROCm and vcpkg runtime DLLs plus
the hipBLASLt/rocBLAS kernel libraries next to `gufo.exe`, so `build\release`
runs as it is. `check.ps1` covers the whole setup: the toolchain, the build,
memory, every model it finds with a fit estimate, and runtime traps.

## License

Gufo's original code is [MIT licensed](LICENSE). Adapted code and dependencies
retain their own notices in [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
Model weights are not bundled and retain their publishers' terms.

## Reference Projects

The initial design is informed by the following open source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for compact model serving, GGUF, and CPU/GPU correctness paths.
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), [Nathanw1014/strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp), and [gaetan-puleo/llama-cpp-strix-halo](https://github.com/gaetan-puleo/llama-cpp-strix-halo) for Strix Halo optimization inspiration.
- [vLLM](https://github.com/vllm-project/vllm) for continuous batching and paged request scheduling.
- [hipEngine](https://github.com/shisa-ai/hipEngine) for torch-free HIP execution, and native speculative-cycle work.
- [ds4](https://github.com/antirez/ds4) for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- [audio.cpp](https://github.com/0xShug0/audio.cpp) for audio models for tts and asr tasks.
