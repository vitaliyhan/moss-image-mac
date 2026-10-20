# Third-party components

This repository contains only glue code (MIT). `setup.sh` downloads the following at pinned
revisions, verifies each file by SHA-256, and stores them locally under `bin/` and `models/`.
Nothing below is redistributed by this repository. Check each upstream license before
commercial use; the licenses listed are as declared by the upstream repos on 2026-10-20.

| Component | Upstream | Pinned revision | License |
|---|---|---|---|
| stable-diffusion.cpp runtime (`sd-server`, `sd-cli`) | [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | release `master-975-95b3090` | MIT |
| MOSS-Image-SFT DiT, 4-bit GGUF (`MOSS-Image-SFT-Q4_0.gguf`) | [leejet/MOSS-Image-SFT-GGUF](https://huggingface.co/leejet/MOSS-Image-SFT-GGUF), quantized from [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | `238daf4` | Apache-2.0 |
| Qwen3.5-9B text encoder, Q4_K_M GGUF | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), from [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `3885219` | Apache-2.0 |
| FLUX.2 small VAE decoder | [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) | `3284a1d` | Apache-2.0 |
| Tokenizer (`tokenizer.json`) | [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | `2c79544` | Apache-2.0 |
| 4-step and 3-step turbo LoRAs (converted to fp16 locally) | [akhaliq/MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps) | `2ded95c` | Apache-2.0 |

Python dependencies (installed into `.venv` from PyPI): `mcp`, `pillow`, `numpy`, `safetensors`.

MOSS-Image citation: *MOSS-Image Technical Report*, arXiv:2610.23374.
