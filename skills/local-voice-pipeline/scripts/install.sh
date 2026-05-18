#!/usr/bin/env bash
# Phase 1 install: brew deps + venv + python pkgs + model downloads.
# Safe to re-run — every step is idempotent.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

echo "===> [1/5] brew packages"
# Note: we use the turboquant fork of llama.cpp from ~/src/llama-cpp-turboquant
# instead of the brew formula — the user has a quantization-tuned binary built
# with Metal support there.
TURBOQUANT_LLAMA_SERVER="$HOME/src/llama-cpp-turboquant/build/bin/llama-server"
if [ ! -x "$TURBOQUANT_LLAMA_SERVER" ]; then
    echo "  ✗ turboquant llama-server not found at $TURBOQUANT_LLAMA_SERVER"
    echo "    build it first: cd ~/src/llama-cpp-turboquant && cmake --build build -j"
    exit 1
fi
echo "  ✓ turboquant llama-server present"

for pkg in espeak-ng portaudio; do
    if brew list "$pkg" --formula >/dev/null 2>&1; then
        echo "  ✓ $pkg already installed"
    else
        echo "  installing $pkg..."
        brew install "$pkg"
    fi
done

echo "===> [2/5] venv at .venv (Python 3.12 via uv)"
if [ ! -d ".venv" ]; then
    uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "===> [3/5] python deps"
# Pipecat with the providers we need (silero VAD, whisper STT, OpenAI-compat LLM,
# generic TTS). NeuTTS install is separate (its own repo).
uv pip install --upgrade \
    "pipecat-ai[silero,whisper,openai,daily]" \
    "faster-whisper>=1.0" \
    "edge-tts>=6.1" \
    "openai>=1.0" \
    "numpy" \
    "soundfile" \
    "huggingface_hub[cli]"

echo "===> [4/5] NeuTTS Air"
# NeuTTS Air ships as a pip package from Neuphonic
if ! python -c "import neutts" 2>/dev/null; then
    uv pip install neutts
else
    echo "  ✓ neutts already installed"
fi

echo "===> [5/5] models"
mkdir -p "$ROOT/models"

# LLM: reuse the existing Qwen3-8B from ~/src/llama-cpp-turboquant/models/Qwen3-8B/
# (saves the 5GB download). Symlink it in so the rest of the scripts have a
# stable local path.
EXISTING_QWEN="$HOME/src/llama-cpp-turboquant/models/Qwen3-8B/Qwen3-8B-UD-Q5_K_XL.gguf"
LOCAL_QWEN="$ROOT/models/Qwen3-8B-UD-Q5_K_XL.gguf"
if [ -f "$EXISTING_QWEN" ]; then
    if [ ! -e "$LOCAL_QWEN" ]; then
        ln -s "$EXISTING_QWEN" "$LOCAL_QWEN"
        echo "  ✓ symlinked Qwen3-8B from turboquant fork"
    else
        echo "  ✓ Qwen3-8B already linked"
    fi
else
    echo "  ✗ Qwen3-8B not found at $EXISTING_QWEN — build the turboquant fork first"
    exit 1
fi

# faster-whisper base model (auto-downloaded by faster-whisper on first use)
mkdir -p "$ROOT/models/whisper"

echo
echo "✅ install complete."
echo
echo "next steps:"
echo "  source $ROOT/.venv/bin/activate"
echo "  ./scripts/smoke-test.py     # verify imports + model loading"
echo "  ./scripts/llama-server.sh   # start the llama.cpp HTTP server (background)"
echo "  ./scripts/pipeline.py       # run the hello-world pipeline"
