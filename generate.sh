#!/usr/bin/env bash
# Generate one image from the command line.
#   ./generate.sh "your prompt" [width] [height] [seed] [quality]
# quality: standard (default, 4-step turbo) | draft (3-step) | high (official 50 steps, slow)
#
# If the backend is already running (./start_server.sh or the MCP server), the request goes
# through it: models stay loaded and no second copy is put in GPU memory. Otherwise sd-cli
# loads the models for this one image (~15 s extra).
set -euo pipefail
cd "$(dirname "$0")"

PROMPT="${1:-a lovely cat holding a sign that says 'MOSS on Mac'}"
W="${2:-768}"
H="${3:-432}"
SEED="${4:--1}"
QUALITY="${5:-standard}"
PORT="${MOSS_PORT:-18765}"
mkdir -p outputs
OUT="outputs/moss_$(date +%Y%m%d_%H%M%S).png"

if curl -s -m 2 "http://127.0.0.1:${PORT}/sdcpp/v1/capabilities" 2>/dev/null | grep -q 'MOSS-Image-SFT'; then
  exec .venv/bin/python moss_client.py "$PROMPT" --width "$W" --height "$H" --seed "$SEED" \
    --quality "$QUALITY" -o "$OUT"
fi

case "$QUALITY" in
  standard|best) ARGS=(--cfg-scale 1.0 --steps 4 --sigmas "1.0,0.91,0.717,0.024,0.0"); PROMPT="${PROMPT}<lora:moss_4step_f16:1>" ;;
  draft)         ARGS=(--cfg-scale 1.0 --steps 3 --sigmas "1.0,0.835,0.024,0.0");     PROMPT="${PROMPT}<lora:moss_3step_f16:1>" ;;
  high)          ARGS=(--cfg-scale 3.0 --steps 50) ;;
  *) echo "error: quality must be standard, draft or high" >&2; exit 1 ;;
esac

bin/sd-cli \
  --diffusion-model models/diffusion_models/MOSS-Image-SFT-Q4_0.gguf \
  --llm models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf \
  --vae models/vae/flux2_ae_small.safetensors \
  --tokenizer models/tokenizers/tokenizer_qwen3.5.json \
  --lora-model-dir models/loras \
  "${ARGS[@]}" --seed "$SEED" -W "$W" -H "$H" \
  -p "$PROMPT" -o "$OUT"

echo "Saved: $OUT"
