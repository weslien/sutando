#!/usr/bin/env bash
# Start llama.cpp's OpenAI-compat HTTP server with the Qwen model.
# Background process; logs to ../logs/llama-server.log.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

# Binary: turboquant fork (quantization-tuned, Metal build).
LLAMA_SERVER="$HOME/src/llama-cpp-turboquant/build/bin/llama-server"
# Model: Qwen3-8B-UD-Q5_K_XL via symlink set up by install.sh.
MODEL="${ROOT}/models/Qwen3-8B-UD-Q5_K_XL.gguf"
PORT="${LLAMA_PORT:-8081}"
LOG="${ROOT}/logs/llama-server.log"

if [ ! -x "$LLAMA_SERVER" ]; then
    echo "ERROR: turboquant llama-server not at $LLAMA_SERVER — build the fork first" >&2
    exit 1
fi
if [ ! -e "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL — run install.sh first" >&2
    exit 1
fi

mkdir -p "${ROOT}/logs"

# Kill any existing server on the same port.
EXISTING_PID=$(lsof -ti:$PORT 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "stopping existing llama-server on :$PORT (pid=$EXISTING_PID)"
    kill "$EXISTING_PID" 2>/dev/null || true
    sleep 1
fi

echo "starting llama-server on :$PORT with Qwen2.5-7B (Metal)..."
echo "  log: $LOG"

# --jinja enables Qwen's chat-template + tool-call format.
# -ngl 99 puts all layers on Metal (M4 Pro has plenty of unified memory).
# -c 8192 sets context (Qwen's native is 32k, but 8k is plenty for voice turns).
nohup "$LLAMA_SERVER" \
    --model "$MODEL" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --jinja \
    -ngl 99 \
    -c 8192 \
    --no-warmup \
    > "$LOG" 2>&1 &

SERVER_PID=$!
echo "  pid: $SERVER_PID"
echo "  health: curl http://127.0.0.1:$PORT/health"
echo
echo "ready when health returns {\"status\": \"ok\"} (usually 5-10s after start)."
