#!/usr/bin/env bash
# Downloads Qwen-Image-2.1 model files for the "Qwen 2.1" UI tab, all through the
# hf-mirror.com mirror (set HF_ENDPOINT to override). Two runtimes are supported:
#
#   MLX  (mflux, Apple-Silicon native - recommended on MacBook Air)
#     base  4/6/8-bit   ~4.4/6.2/8.0 GB each  (text encoder + transformer + VAE included)
#     turbo 4/6/8-bit   same sizes, 8-step distilled
#   GGUF (stable-diffusion.cpp sd-server, uncensored "UC" and base variants)
#     Q4_0 / Q4_K_M / Q5_K_M / Q6_K / Q8_0  ~4-7.6 GB each
#     + shared text encoder (~9.4 GB), VAE (~0.7 GB) and tokenizer
#
# Usage:
#   ./setup_qwen.sh                    # interactive: pick what to download
#   ./setup_qwen.sh mlx-q4             # small MLX base 4-bit (the recommended first model)
#   ./setup_qwen.sh mlx-turbo-q4 gguf-uc-q4_K_M gguf-enc
#   ./setup_qwen.sh small-encoder      # optional experimental fast prompt encoder (MLX)
#
# Safe to re-run: finished downloads are skipped, partial ones resume.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

HF="${HF_ENDPOINT:-https://hf-mirror.com}"
QWEN_REPO="abenzerps/Qwen-Image-2.1-Uncensored-Quantized"   # GGUF + text encoder + VAE
MLX_REPO="OsaurusAI"                                        # mflux-format MLX repos

fetch() {  # dest url  - retries on rate limits (429) and dropped connections
  local dest=$1 url=$2
  if [[ -s "$dest" ]]; then
    echo "   ok  $dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "   get $dest"
  local attempt
  for attempt in 1 2 3 4 5 6; do
    if curl -fL --retry 3 --retry-delay 5 -C - --progress-bar -o "$dest.part" "$url"; then
      mv "$dest.part" "$dest"
      return
    fi
    echo "   attempt $attempt failed (mirror rate limit?) - waiting 60 s"
    sleep 60
  done
  rm -f "$dest.part"
  die "download kept failing: $url"
}

fetch_repo() {  # local-dir hf-repo  (mflux-format: transformer/, text_encoder/, vae/, processor/)
  local dest=$1 repo=$2
  local ok=1
  for f in model_index.json transformer/model.safetensors.index.json \
           text_encoder/model.safetensors.index.json vae/model.safetensors.index.json; do
    [[ -s "$dest/$f" ]] || ok=0
  done
  if (( ok )); then
    echo "   ok  $dest"
    return
  fi
  say "MLX weights: $repo (~4-8 GB, all components included)"
  mkdir -p "$dest"
  while read -r f; do
    [[ "$f" == .* ]] && continue
    fetch "$dest/$f" "$HF/$repo/resolve/main/$f"
  done < <(curl -fsSL "$HF/api/models/$repo" | python3 -c '
import json, sys
for s in json.load(sys.stdin).get("siblings", []):
    n = s["rfilename"]
    if n.endswith((".json", ".jinja", ".safetensors")):
        print(n)')
}

# ------------------------------------------------------------------ catalog
MLX_VARIANTS=(
  "mlx-q4|Qwen-Image-2.1-mflux-4bit"
  "mlx-q6|Qwen-Image-2.1-mflux-6bit"
  "mlx-q8|Qwen-Image-2.1-mflux-8bit"
  "mlx-turbo-q4|Qwen-Image-2.1-Turbo-mflux-4bit"
  "mlx-turbo-q6|Qwen-Image-2.1-Turbo-mflux-6bit"
  "mlx-turbo-q8|Qwen-Image-2.1-Turbo-mflux-8bit"
)
GGUF_QUANTS=(Q4_0 Q4_K_M Q5_K_M Q6_K Q8_0)

gguf_uc()    { echo "models/diffusion_models/qwen-image-2.1-UC-$1.gguf|$HF/$QWEN_REPO/resolve/main/qwen-image-2.1-UC-$1.gguf"; }
gguf_base()  { echo "models/diffusion_models/qwen-image-2.1-$1.gguf|$HF/$QWEN_REPO/resolve/base/qwen-image-2.1-$1.gguf"; }
# Text encoder: the int8 "convrot" safetensors from abenzerps crashes this sd-server build
# (missing Metal kernel kernel_mul_mm_i8_i8), so we use the GGUF Q8_0 of the same Qwen3-VL-8B.
GGUF_ENC="models/text_encoders/Qwen3-VL-8B-Instruct-Q8_0.gguf|$HF/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-Q8_0.gguf"
# The VAE comes from the official Qwen/Qwen-Image-2.1 repo (diffusers format): the copy in
# the abenzerps repo 404s on hf-mirror with "Entry not found" even though it is listed.
GGUF_VAE="models/vae/qwen_image_2_1_vae.safetensors|$HF/Qwen/Qwen-Image-2.1/resolve/main/vae/diffusion_pytorch_model.safetensors"
GGUF_TOK="models/tokenizers/tokenizer_qwen3vl.json|$HF/$MLX_REPO/Qwen-Image-2.1-mflux-4bit/resolve/main/processor/tokenizer.json"
# 2-step DMD2 distillation LoRA (linoyts), applied on top of the MLX base weights.
DMD2_LORA="models/lora/qwen-image-2.1-dmd2-2steps.safetensors"

download_one() {
  local c=$1
  case "$c" in
    mlx-*)
      for v in "${MLX_VARIANTS[@]}"; do
        if [[ "$v" == "$c|"* ]]; then
          fetch_repo "models/mlx/${v%%|*}" "$MLX_REPO/${v##*|}"
          return
        fi
      done
      die "unknown MLX choice: $c"
      ;;
    gguf-enc)
      say "GGUF text encoder (~9.4 GB, shared by all GGUF quants)"
      fetch ${GGUF_ENC%%|*} "${GGUF_ENC##*|}"
      ;;
    gguf-vae)
      say "GGUF VAE (~0.7 GB, shared)"
      fetch ${GGUF_VAE%%|*} "${GGUF_VAE##*|}"
      ;;
    gguf-tok)
      say "GGUF tokenizer (shared)"
      fetch ${GGUF_TOK%%|*} "${GGUF_TOK##*|}"
      ;;
    gguf-uc-*)
      local q="$(echo "${c#gguf-uc-}" | tr '[:lower:]' '[:upper:]')"
      [[ " ${GGUF_QUANTS[*]} " == *" $q "* ]] || die "unknown quant: $q"
      local line; line="$(gguf_uc "$q")"
      say "GGUF uncensored $q (~4-8 GB)"
      fetch "${line%%|*}" "${line##*|}"
      ;;
    gguf-base-*)
      local q="$(echo "${c#gguf-base-}" | tr '[:lower:]' '[:upper:]')"
      [[ " ${GGUF_QUANTS[*]} " == *" $q "* ]] || die "unknown quant: $q"
      local line; line="$(gguf_base "$q")"
      say "GGUF base $q (~4-8 GB)"
      fetch "${line%%|*}" "${line##*|}"
      ;;
    small-encoder)
      say "Small MLX prompt encoder (~2.1 GB, experimental: English prompts only)"
      ensure_mflux
      HF_ENDPOINT="$HF" .venv/bin/python - <<'PY'
from mflux.models.qwen21.model.qwen21_small_text_encoder.zen_image_edit_weights import ZenImageEditWeights
print("   ok ", ZenImageEditWeights.snapshot())
PY
      ;;
    lora-dmd2)
      say "DMD2 2-step LoRA for MLX Qwen-Image-2.1 (~64 MB)"
      ensure_mflux
      fetch "$DMD2_LORA" "$HF/linoyts/Qwen-Image-2.1-DMD2-2steps/resolve/main/pytorch_lora_weights.safetensors"
      # mflux expects PEFT-style lora_A/lora_B keys; the LoRA ships old-style lora.down/up.
      HF_ENDPOINT="$HF" .venv/bin/python - <<'PY'
import mlx.core as mx
path = "models/lora/qwen-image-2.1-dmd2-2steps.safetensors"
w = mx.load(path)
if any(".lora.down." in k for k in w):
    w = {k.removeprefix("transformer.")
          .replace(".lora.down.", ".lora_A.default.")
          .replace(".lora.up.", ".lora_B.default."): v for k, v in w.items()}
    mx.save_safetensors(path, w)
    print("   converted keys to PEFT style")
else:
    print("   ok  already PEFT-style")
PY
      ;;
    *) die "unknown choice: $c" ;;
  esac
}

# ------------------------------------------------------------------ python env (mflux)
ensure_mflux() {
  [[ -x .venv/bin/python ]] || die ".venv missing - run ./setup.sh first"
  if .venv/bin/python -c 'import mflux' 2>/dev/null; then
    echo "   ok  mflux installed"
  else
    say "Installing mflux (MLX runtime) into .venv"
    if command -v uv >/dev/null; then
      uv pip install -q -p .venv/bin/python mflux
    else
      .venv/bin/python -m pip install -q mflux
    fi
  fi
}

# ------------------------------------------------------------------ choose
CHOICES=("$@")
if (( ${#CHOICES[@]} == 0 )); then
  say "Available models (all downloads via $HF)"
  echo "  MLX (mflux runtime, recommended on MacBook Air):"
  echo "    mlx-q4 / mlx-q6 / mlx-q8           base, 40 steps (~4.4/6.2/8.0 GB)"
  echo "    mlx-turbo-q4 / -q6 / -q8           turbo, 8 steps, same sizes"
  echo "    small-encoder                      experimental fast prompt encoder (~2.1 GB)"
  echo "    lora-dmd2                          DMD2 2-step LoRA (~64 MB, base model only)"
  echo "  GGUF (sd-server runtime):"
  echo "    gguf-uc-Q4_0|Q4_K_M|Q5_K_M|Q6_K|Q8_0   uncensored edit model"
  echo "    gguf-base-Q4_0|...|Q8_0                official base"
  echo "    gguf-enc / gguf-vae / gguf-tok         shared GGUF components (required once)"
  echo
  read -rp "Choices (space-separated, e.g. 'mlx-q4 gguf-uc-Q4_K_M gguf-enc gguf-vae gguf-tok'): " line
  read -ra CHOICES <<< "$line"
  (( ${#CHOICES[@]} > 0 )) || die "nothing selected"
fi

for c in "${CHOICES[@]}"; do download_one "$c"; done
[[ "${CHOICES[*]}" == *mlx* || "${CHOICES[*]}" == *small-encoder* ]] && ensure_mflux
say "Done. Start the Qwen tab from the UI, or for GGUF: MOSS_SERVER_MODEL=qwen2.1 ./start_server.sh"
