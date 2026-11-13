#!/usr/bin/env bash
# Adds "Local LLM Sandbox" to your app menu so you can start it with one click.
#
#   ./install.sh               add the app-menu launcher
#   ./install.sh --deps        also install system packages and a private Python environment (.venv) first
#                              (Pop!_OS, Ubuntu, Debian, Fedora)
#   ./install.sh --uninstall   remove the launcher (your chats, models and .venv are left alone)
set -euo pipefail

APP_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ENTRY_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ENTRY="$ENTRY_DIR/local-llm-sandbox.desktop"
VENV="$APP_DIR/.venv"

if [[ "${1:-}" == "--uninstall" ]]; then
    rm -f "$ENTRY"
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$ENTRY_DIR" 2>/dev/null || true
    echo "Removed $ENTRY"
    exit 0
fi

if [[ "${1:-}" == "--deps" ]]; then
    echo "Installing system packages (you may be asked for your password)..."
    if command -v apt-get >/dev/null 2>&1; then
        # A broken third-party repository (a 404 PPA, say) makes update exit with an error; that shouldn't stop us.
        sudo apt-get update || echo "(apt reported a problem with one of your repositories; continuing anyway)"
        if ! sudo apt-get install -y espeak-ng python3.12 python3.12-venv ffmpeg curl build-essential; then
            echo
            echo "Couldn't install Python 3.12 from your repositories (older Pop!_OS and Ubuntu releases don't have it)."
            echo "Add it from the deadsnakes PPA, then run this again:"
            echo "  sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update"
            exit 1
        fi
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y espeak-ng python3.12 gcc gcc-c++ ffmpeg-free curl
    else
        echo "Couldn't find apt-get or dnf. Install espeak-ng, Python 3.12 and ffmpeg yourself, then run this again."
        exit 1
    fi

    # A private environment keeps these packages away from the system Python
    # (newer Ubuntu and Pop!_OS block system-wide pip installs).
    echo "Creating a private Python environment in $VENV ..."
    python3.12 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    echo "Installing Python packages (this is a big download, a few GB) ..."
    "$VENV/bin/python" -m pip install -r "$APP_DIR/requirements.txt"
    "$VENV/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    "$VENV/bin/python" -m pip install kokoro

    if ! command -v ollama >/dev/null 2>&1; then
        echo
        echo "Ollama isn't installed yet. Get it from https://ollama.com, then pull a model, for example: ollama pull llama3.2"
    fi
fi

chmod +x "$APP_DIR/launch.sh"
mkdir -p "$ENTRY_DIR"
cat > "$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Name=Local LLM Sandbox
Comment=Private local AI chat with voice, files and photos
Exec="$APP_DIR/launch.sh"
Icon=$APP_DIR/assets/icon.svg
Terminal=false
Categories=Utility;Development;
StartupNotify=true
StartupWMClass=LocalLLMSandbox
EOF
chmod +x "$ENTRY"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$ENTRY_DIR" 2>/dev/null || true

echo
echo "Done. Open your app menu (press the Super key and type \"Local LLM\") and click Local LLM Sandbox."
echo "To pin it, right-click the icon while it's open and choose Add to Favorites, or drag it onto your dock."
echo "If something fails to start, run ./launch.sh in a terminal, or read ${XDG_CACHE_HOME:-$HOME/.cache}/local-llm-sandbox/streamlit.log"
