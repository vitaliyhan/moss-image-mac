#!/usr/bin/env python3
"""Convert a turbo LoRA's fp32 matrices to fp16 (alphas stay fp32).

Halves the file (1.19 GB -> 0.59 GB) and cuts per-step time ~20% on Apple GPUs. The LoRA
is still applied unmerged at runtime, so the distillation delta is kept separate from the
4-bit base weights (merging it into low-precision weights is what damages fine text).

usage: convert_lora_fp16.py <in.safetensors> <out.safetensors>
"""
import sys

import numpy as np
from safetensors.numpy import load_file, save_file


def main(src, dst):
    tensors = load_file(src)
    out = {k: (v.astype(np.float16) if v.ndim == 2 else v) for k, v in tensors.items()}
    peak = max(float(np.abs(v).max()) for v in tensors.values() if v.ndim == 2)
    if peak > 6.0e4:
        raise SystemExit(f"value {peak} exceeds fp16 range; keep this LoRA in fp32")
    save_file(out, dst)
    print(f"{dst}: {len(out)} tensors, max |w| = {peak:.4g}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
