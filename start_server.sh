#!/usr/bin/env bash
# Persistent MOSS-Image-SFT INT4 backend (stable-diffusion.cpp sd-server + Metal).
# Keeps the 4-bit DiT, Qwen3.5-9B text encoder and VAE loaded so each request only pays for
# sampling. The MCP server starts this automatically; run it yourself for HTTP/CLI use.
#
#   http://127.0.0.1:18765/sdcpp/v1/img_gen      native async API (txt2img / img2img / inpaint)
#   http://127.0.0.1:18765/sdapi/v1/txt2img      A1111-compatible
#   http://127.0.0.1:18765/v1/images/generations OpenAI-compatible
#
# Env: MOSS_PORT (default 18765), MOSS_HOST (default 127.0.0.1 - keep it local unless you
# know you want the API on your network; it has no authentication).
#
# Model selection (the Qwen 2.1 tab uses these too):
#   MOSS_SERVER_MODEL=moss|qwen2.1   which model to load (default moss)
#   QWEN_VARIANT=uc|base             qwen2.1 only: uncensored (default) or official base
#   QWEN_QUANT=Q4_0|Q4_K_M|Q5_K_M|Q6_K|Q8_0   qwen2.1 GGUF quant (default Q4_K_M)
set -euo pipefail
cd "$(dirname "$0")"

PORT="${MOSS_PORT:-18765}"
HOST="${MOSS_HOST:-127.0.0.1}"
SERVER_MODEL="${MOSS_SERVER_MODEL:-moss}"
QWEN_VARIANT="${QWEN_VARIANT:-uc}"
QWEN_QUANT="${QWEN_QUANT:-Q4_K_M}"

if [[ "$SERVER_MODEL" == "moss" ]]; then
  FILES=(bin/sd-server models/diffusion_models/MOSS-Image-SFT-Q4_0.gguf
         models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf models/vae/flux2_ae_small.safetensors
         models/tokenizers/tokenizer_qwen3.5.json)
elif [[ "$SERVER_MODEL" == "qwen2.1" ]]; then
  if [[ "$QWEN_VARIANT" == "uc" ]]; then
    QWEN_GGUF="models/diffusion_models/qwen-image-2.1-UC-${QWEN_QUANT}.gguf"
  elif [[ "$QWEN_VARIANT" == "base" ]]; then
    QWEN_GGUF="models/diffusion_models/qwen-image-2.1-${QWEN_QUANT}.gguf"
  else
    echo "error: QWEN_VARIANT must be uc or base (got '$QWEN_VARIANT')" >&2; exit 1
  fi
  FILES=(bin/sd-server "$QWEN_GGUF"
         models/text_encoders/Qwen3-VL-8B-Instruct-Q8_0.gguf
         models/vae/qwen_image_2_1_vae.safetensors
         models/tokenizers/tokenizer_qwen3vl.json)
else
  echo "error: MOSS_SERVER_MODEL must be moss or qwen2.1 (got '$SERVER_MODEL')" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  [[ -e "$f" ]] || { echo "error: $f missing - run ./setup.sh (moss) or ./setup_qwen.sh (qwen2.1) first" >&2; exit 1; }
done

# Never load a second copy of the models: two backends exhaust GPU memory and produce
# blank images.
if curl -s -m 2 "http://127.0.0.1:${PORT}/sdcpp/v1/capabilities" 2>/dev/null | grep -q '"model"'; then
  echo "A backend is already running on port ${PORT}; nothing to do." >&2
  exit 0
fi
if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "error: port ${PORT} is used by another program; set MOSS_PORT to a free port." >&2
  exit 1
fi

if [[ "$SERVER_MODEL" == "moss" ]]; then
  exec bin/sd-server \
    --diffusion-model models/diffusion_models/MOSS-Image-SFT-Q4_0.gguf \
    --llm models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf \
    --vae models/vae/flux2_ae_small.safetensors \
    --tokenizer models/tokenizers/tokenizer_qwen3.5.json \
    --lora-model-dir models/loras \
    --listen-ip "$HOST" \
    --listen-port "$PORT" \
    "$@"
else
  ARGS=(--diffusion-model "$QWEN_GGUF"
        --llm models/text_encoders/Qwen3-VL-8B-Instruct-Q8_0.gguf
        --vae models/vae/qwen_image_2_1_vae.safetensors
        --tokenizer models/tokenizers/tokenizer_qwen3vl.json
        --model-args qwen_image_2_1_prefix_cache=1
        --listen-ip "$HOST"
        --listen-port "$PORT")
  exec bin/sd-server "${ARGS[@]}" "$@"
fi
