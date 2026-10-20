# Performance notes

All numbers were measured on a **Mac mini M4 (10-core GPU, 24 GB), macOS 26**, with stable-diffusion.cpp
`master-975-95b3090` on Metal and a warm backend unless stated otherwise.

## Where the time goes (standard mode, 768×432)

| Stage | Time |
|---|---|
| Text encoding (Qwen3.5-9B Q4_K_M), new prompt | ~0.9 s (cached prompts: ~0 s) |
| DiT sampling, 4 steps × ~7 s | ~28-30 s |
| VAE decode (FLUX.2 small decoder) | ~3 s |
| **Total** | **~30-35 s** |

A forward pass of the 5B DiT costs about 15 TFLOP at 768×432, since text tokens join the image sequence. The base M4
GPU peaks at ~4 TFLOPS on paper and sustains ~2.4-2.9 TFLOPS on these matrix shapes in MLX benchmarks on the same
machine. Four passes therefore need **at least ~14 s even at theoretical peak, and ~22 s at realistic efficiency**,
before decoding. The current ~28-30 s is within ~1.3× of that realistic floor.

## Optimisations in use

| Change | Effect | Quality |
|---|---|---|
| 4-step DMD turbo LoRA (akhaliq v2.0) instead of 50 steps with CFG | 100 → 4 forward passes (~20× faster) | Comparable; text and layout as good or better (see README) |
| LoRA in fp16 instead of fp32 (applied unmerged at runtime) | ~7.7 → ~6.1 s per pass at 512² | Identical outputs (values well inside fp16 range) |
| Q4_0 GGUF instead of Q4_K | ~1.5 s faster per image at the same size | Same bits per weight |
| FLUX.2 small decoder | Faster decode, and it's what the official repo ships | Official component |
| Persistent backend (`sd-server`) | Avoids ~15 s model load per image | — |
| Snapped release sigmas `1.0, 0.91, 0.717, 0.024` | Matches the LoRA's training anchors exactly | Required for correct output |
| Step previews only in `high` mode | Avoids ~1% overhead on turbo jobs | — |

## Tried and rejected

| Idea | Result |
|---|---|
| EasyCache / Spectrum step caching (30 steps) | 333 s → 165-205 s, still far slower than turbo, and step caching gives ~nothing at 4 steps |
| Merging the LoRA into the Q4 weights at load (`--lora-apply-mode immediately`) | Crashes on quantized weights; merging also destroys the distillation delta |
| Fusing the q/k/v LoRA pairs into one (mathematically identical) | 1.4 s **slower** (32.6 vs 31.1 s) |
| RealESRGAN x4 upscaling | ~80 s per image on this GPU |
| MLX ports (mflux, moss-image-mlx) | Quantized MLX is not faster than sd.cpp on M-series per their benchmarks; bf16 needs ~10 GB more memory |
| Draw Things | Doesn't support MOSS-Image (Oct 2026) |
| 2-step LoRA | Author-documented failure: blurred or duplicated subjects, garbled small text |
| TAEF2 tiny VAE, 3-step LoRA, lower resolution, DBCache | Each saves 2-16 s but visibly costs detail or text, so they're not used (3-step is exposed as `draft`) |
| Apple Neural Engine (Core ML) | No 5B DiT has been shown running on it; FP16-only numerics; unproven |

Bottom line: getting under ~10 s per image without losing quality isn't possible on a base M4. M5-generation chips
(with matrix units in the GPU) are reported to be 3.5-4× faster on this kind of workload.

## `high` mode

`high` runs 50 steps with CFG 3, so two passes per step: ~12.3 s per step at 768×432, about **10.4 min per image**.
It uses the same memory as `standard`. Its output was checked to be pixel-identical (mean absolute difference 0.0)
to a reference `sd-cli` run with the same seed and settings.
