#!/usr/bin/env bash
# One-click launcher: starts Ollama if it isn't running, starts the sandbox, and opens it.
# Safe to run again while it's already up: it just opens the window.

APP_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PORT="${LLM_SANDBOX_PORT:-8501}"
URL="http://localhost:${PORT}"
LOG_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/local-llm-sandbox"
mkdir -p "$LOG_DIR"

say() {
    echo "$1"
    command -v notify-send >/dev/null 2>&1 && notify-send -a "Local LLM Sandbox" "Local LLM Sandbox" "$1"
}

healthy() { curl -fs --max-time 2 "$1" >/dev/null 2>&1; }

# Prefer the private environment that install.sh --deps creates. Kokoro (voice) needs Python 3.12.
if [ -x "$APP_DIR/.venv/bin/python" ]; then
    PY="$APP_DIR/.venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
    PY=python3.12
else
    PY=python3
fi

if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
    say "The Python packages aren't installed yet. Open a terminal in $APP_DIR and run: ./install.sh --deps"
    exit 1
fi

# 1. Ollama
if ! healthy "http://127.0.0.1:11434/api/tags"; then
    if command -v ollama >/dev/null 2>&1; then
        setsid nohup ollama serve >"$LOG_DIR/ollama.log" 2>&1 &
        for _ in $(seq 1 30); do
            healthy "http://127.0.0.1:11434/api/tags" && break
            sleep 0.5
        done
    else
        say "Ollama isn't installed. Get it from https://ollama.com (the app will still open)."
    fi
fi

# 2. The sandbox itself
if ! healthy "$URL/_stcore/health"; then
    cd "$APP_DIR" || exit 1
    setsid nohup "$PY" -m streamlit run app.py \
        --server.headless true \
        --server.address 127.0.0.1 \
        --server.port "$PORT" \
        --browser.gatherUsageStats false \
        --theme.base dark \
        >"$LOG_DIR/streamlit.log" 2>&1 &
    for _ in $(seq 1 120); do
        healthy "$URL/_stcore/health" && break
        sleep 0.5
    done
    if ! healthy "$URL/_stcore/health"; then
        say "The sandbox didn't start. Details: $LOG_DIR/streamlit.log"
        exit 1
    fi
fi

# 3. Open it. Chromium-based browsers get their own app-style window, anything else uses your default browser.
for browser in google-chrome google-chrome-stable chromium chromium-browser brave-browser microsoft-edge; do
    if command -v "$browser" >/dev/null 2>&1; then
        exec "$browser" --app="$URL" --class=LocalLLMSandbox
    fi
done
exec xdg-open "$URL"
