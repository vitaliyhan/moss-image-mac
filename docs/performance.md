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

`high` runs 50 steps with CFG 3, so two passes per step: ~12.3 s per step at 768×432 (about **10.4 min per
image**) and ~25.6 s per step at 1024×576 (about **21.3 min**).
It uses the same memory as `standard`. Its output was checked to be pixel-identical (mean absolute difference 0.0)
to a reference `sd-cli` run with the same seed and settings.

## Qwen 2.1 tab (MLX / mflux)

Measured on an Apple-Silicon Mac, 24 GB, mflux 0.24 / mlx 0.32, Qwen-Image-2.1 base 4-bit, 512×512, 20 steps:

| Path | Per image |
|---|---|
| mflux CLI (old: one subprocess per render) | ~69 s (spawn + import + load + render) |
| Persistent worker, first render | ~69 s (spawn + load are ~3 s; weights are page-cached and quantization is lazy) |
| Persistent worker, subsequent renders | ~68 s idle; back-to-back renders throttle to ~83 s and recover after ~75 s GPU idle |

Notes:

- The worker (`qwen_mlx_server.py`, spawned on demand by `qwen_mlx.py`) keeps weights, compiled graphs and the
  transformer's text/image-prefix K/V caches resident across renders. The saving is modest at small render sizes
  (~5-7 s of per-render CLI overhead) and grows with batch jobs, which previously paid it per image.
- Worker output is pixel-identical to the CLI (same seed, max pixel diff 0). Cancel mid-render leaves the worker
  reusable; the next render is again pixel-identical.
- Sustained back-to-back rendering slows down ~17% from GPU boost-clock throttling regardless of runtime; short
  idle gaps between jobs recover it. Not a worker regression.
- "Fast prompt encoder" (experimental, UI toggle): swaps the Qwen3-VL-8B encoder for Qwen3.5-0.8B + fusion adapter
  (~2.1 GB instead of ~17.5 GB download). English prompts only; numerals in rendered text can be wrong.
  Pre-download with `./setup_qwen.sh small-encoder`.
- **DMD2 LoRA** (UI toggle "DMD2 — 3-step turbo"): a [DMD2 distillation LoRA](https://hf-mirror.com/linoyts/Qwen-Image-2.1-DMD2-2steps)
  applied on top of the base weights at load. Trained without CFG; the toggle turns CFG off and runs **3 steps**
  (2 is the trained minimum, 3 gives a finished image). ~5-8 s per 512² image warm (vs ~32 s for 10 steps × CFG 2.5)
  at slightly softer detail. Toggling it reloads the resident model (~3 s + ~3 s LoRA bake).
  Download: `./setup_qwen.sh lora-dmd2`.
- At resolutions above 1024² pixel area the worker tiles the VAE decode (512 px tiles) to stay off swap.
- `MOSS_MLX_SUBPROCESS=1` bypasses the worker and shells out to the mflux CLI per render, the old way.
  Worker lifecycle: spawned lazily, port in `logs/mlx_worker.json`, log in `logs/mlx_worker.log`; it exits when
  the app that started it exits.
