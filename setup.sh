#!/usr/bin/env bash
# One-time setup for moss-image-mac: downloads the stable-diffusion.cpp runtime and the
# INT4 MOSS-Image-SFT model files (~9.9 GB), verifies every file by SHA-256, converts the
# turbo LoRAs to fp16 and creates the Python environment for the MCP server.
#
# Safe to re-run: finished downloads are verified and skipped, partial ones resume.
# usage: ./setup.sh [--build-from-source]
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

BUILD_FROM_SOURCE=0
[[ "${1:-}" == "--build-from-source" ]] && BUILD_FROM_SOURCE=1

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------- pinned inputs
SDCPP_TAG="master-975-95b3090"
SDCPP_COMMIT="95b3090"
SDCPP_ZIP="sd-master-95b3090-bin-Darwin-macOS-26.6.2-arm64.zip"
SDCPP_SHA="2edf5a21c4b73f448c3a02c87925283b77aa94f1859c0d8e649134da68b40813"

# Downloads go through the hf-mirror.com mirror by default; set HF_ENDPOINT to override
# (e.g. HF_ENDPOINT=https://huggingface.co ./setup.sh).
HF="${HF_ENDPOINT:-https://hf-mirror.com}"
# dest | url | sha256
MODELS=(
  "models/diffusion_models/MOSS-Image-SFT-Q4_0.gguf|$HF/leejet/MOSS-Image-SFT-GGUF/resolve/238daf423f0de65316207453591f275407aac9a7/MOSS-Image-SFT-Q4_0.gguf|b59d2461fa431d670471f74ab54fd90a0bfb36799fd3ffbec1cf103568a883b8"
  "models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf|$HF/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf|03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"
  "models/vae/flux2_ae_small.safetensors|$HF/Comfy-Org/MOSS-Image/resolve/3284a1dbb351e3396d70f39f2a71cbf6ed2ef1bb/vae/flux2-vae_small.safetensors|d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b"
  "models/tokenizers/tokenizer_qwen3.5.json|$HF/OpenMOSS-Team/MOSS-Image-SFT/resolve/2c7954437269461767f533b006823e8f85973a60/tokenizer/tokenizer.json|06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
)
# fp32 source LoRA | url | sha256 | fp16 output
LORA_REV="2ded95c37a029360245926fe72e475e30039bbfb"
LORAS=(
  "models/loras/.src_4step.safetensors|$HF/akhaliq/MOSS-Image-SFT-4-steps/resolve/$LORA_REV/moss_image_sft_4step_v2.0.safetensors|154c7f1455c3bf9925945c2f1ea2e4f09d0d0c67256081384a094b2b585f913f|models/loras/moss_4step_f16.safetensors"
  "models/loras/.src_3step.safetensors|$HF/akhaliq/MOSS-Image-SFT-4-steps/resolve/$LORA_REV/moss_image_sft_3step_v1.0.safetensors|cde8e7e8cadab93c7081cca1892bc73aaced8bdb583c7dfb9903d2f4e66aa4b2|models/loras/moss_3step_f16.safetensors"
)

# ---------------------------------------------------------------------------- preflight
[[ "$(uname -s)" == "Darwin" ]] || die "macOS only (Metal backend)."
[[ "$(uname -m)" == "arm64" ]] || die "Apple Silicon (M1 or newer) required."
MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
FREE_GB=$(( $(df -k "$ROOT" | awk 'NR==2 {print $4}') / 1048576 ))
say "macOS $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), ${MEM_GB} GB RAM, ${FREE_GB} GB free"
(( MEM_GB >= 16 )) || warn "16 GB+ unified memory recommended (peak use is ~9 GB)."
(( FREE_GB >= 14 )) || warn "~13 GB of free disk is needed during setup (11 GB after)."
command -v curl >/dev/null || die "curl not found."

verify() {  # file sha256 -> 0 if file exists and matches
  [[ -f "$1" ]] && [[ "$(shasum -a 256 "$1" | cut -d' ' -f1)" == "$2" ]]
}

fetch() {  # dest url sha256
  local dest=$1 url=$2 sha=$3
  if verify "$dest" "$sha"; then
    echo "   ok  $dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "   get $dest"
  curl -fL --retry 5 --retry-delay 3 -C - --progress-bar -o "$dest.part" "$url" \
    || curl -fL --retry 5 --retry-delay 3 --progress-bar -o "$dest.part" "$url"
  [[ "$(shasum -a 256 "$dest.part" | cut -d' ' -f1)" == "$sha" ]] \
    || { rm -f "$dest.part"; die "checksum mismatch for $dest (deleted, re-run to retry)"; }
  mv "$dest.part" "$dest"
}

# ---------------------------------------------------------------------------- runtime
say "stable-diffusion.cpp runtime ($SDCPP_TAG)"
if [[ -x bin/sd-server && -x bin/sd-cli ]]; then
  echo "   ok  bin/sd-server, bin/sd-cli"
elif (( BUILD_FROM_SOURCE == 0 && MACOS_MAJOR >= 26 )); then
  mkdir -p bin
  fetch "bin/$SDCPP_ZIP" "https://github.com/leejet/stable-diffusion.cpp/releases/download/$SDCPP_TAG/$SDCPP_ZIP" "$SDCPP_SHA"
  (cd bin && unzip -oq "$SDCPP_ZIP" && rm -f "$SDCPP_ZIP")
  xattr -dr com.apple.quarantine bin 2>/dev/null || true
else
  (( MACOS_MAJOR >= 26 )) || say "Prebuilt binaries need macOS 26+; building from source instead."
  command -v cmake >/dev/null || die "cmake required to build (brew install cmake)."
  xcode-select -p >/dev/null 2>&1 || die "Xcode Command Line Tools required (xcode-select --install)."
  SRC="$ROOT/.build/stable-diffusion.cpp"
  if [[ ! -d "$SRC/.git" ]]; then
    git clone --recursive https://github.com/leejet/stable-diffusion.cpp "$SRC"
  fi
  git -C "$SRC" fetch --tags --quiet
  git -C "$SRC" checkout --quiet "$SDCPP_COMMIT"
  git -C "$SRC" submodule update --init --recursive --quiet
  cmake -S "$SRC" -B "$SRC/build" -DSD_METAL=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$SRC/build" --config Release -j "$(sysctl -n hw.ncpu)"
  mkdir -p bin
  find "$SRC/build" \( -name sd-cli -o -name sd-server -o -name 'libstable-diffusion*.dylib' \) -type f \
    -exec cp {} bin/ \;
  [[ -x bin/sd-server ]] || die "build finished but sd-server was not found under $SRC/build"
fi
bin/sd-cli --help >/dev/null 2>&1 || die "bin/sd-cli does not run on this Mac."

# ---------------------------------------------------------------------------- models
say "Model files (~9.3 GB; verified by SHA-256)"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r dest url sha <<<"$entry"
  fetch "$dest" "$url" "$sha"
done

# ---------------------------------------------------------------------------- python env
say "Python environment (.venv)"
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null; then
    uv venv .venv -p 3.12 -q
  else
    PY=""
    for c in python3.13 python3.12 python3.11 python3.10 python3; do
      if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
        PY="$c"; break
      fi
    done
    [[ -n "$PY" ]] || die "Python 3.10+ required (brew install python@3.12, or install uv)."
    "$PY" -m venv .venv
  fi
fi
if command -v uv >/dev/null; then
  uv pip install -q -p .venv/bin/python -r requirements.txt
else
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
fi

# ---------------------------------------------------------------------------- turbo LoRAs
say "Turbo LoRAs (download fp32, convert to fp16)"
for entry in "${LORAS[@]}"; do
  IFS='|' read -r src url sha out <<<"$entry"
  if [[ -f "$out" ]]; then
    echo "   ok  $out"
    continue
  fi
  fetch "$src" "$url" "$sha"
  .venv/bin/python scripts/convert_lora_fp16.py "$src" "$out"
  rm -f "$src"
done

# ---------------------------------------------------------------------------- done
chmod +x start_server.sh generate.sh 2>/dev/null || true
mkdir -p outputs logs
say "Setup complete."
cat <<EOF

Try it:
  ./generate.sh "a lovely cat holding a sign that says 'hello mac'"

Add the MCP server to your app (Claude Desktop, Cursor, VS Code, ...):
{
  "mcpServers": {
    "moss-image": {
      "command": "$ROOT/.venv/bin/python",
      "args": ["$ROOT/moss_mcp.py"]
    }
  }
}

Claude Code:
  claude mcp add moss-image -s user -- "$ROOT/.venv/bin/python" "$ROOT/moss_mcp.py"

EOF
