#!/usr/bin/env bash
# Adds "Local LLM Sandbox" to your app menu so you can start it with one click.
#
#   ./install.sh            add the app-menu launcher
#   ./install.sh --deps     also install system and Python packages first (Fedora or Ubuntu/Debian)
#   ./install.sh --uninstall   remove the launcher (your chats and models are left alone)
set -euo pipefail

APP_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ENTRY_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ENTRY="$ENTRY_DIR/local-llm-sandbox.desktop"

if [[ "${1:-}" == "--uninstall" ]]; then
    rm -f "$ENTRY"
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$ENTRY_DIR" 2>/dev/null || true
    echo "Removed $ENTRY"
    exit 0
fi

if [[ "${1:-}" == "--deps" ]]; then
    echo "Installing system packages (you may be asked for your password)..."
    if command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y espeak-ng python3.12 gcc gcc-c++ ffmpeg-free curl
    elif command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y espeak-ng python3.12 python3.12-venv ffmpeg curl
    else
        echo "Couldn't find dnf or apt-get. Install espeak-ng, Python 3.12 and ffmpeg yourself, then run this again."
        exit 1
    fi

    echo "Installing Python packages for Python 3.12..."
    python3.12 -m ensurepip --user >/dev/null 2>&1 || true
    python3.12 -m pip install --user -U pip
    python3.12 -m pip install --user -r "$APP_DIR/requirements.txt"
    python3.12 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
    python3.12 -m pip install --user kokoro

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
echo "If something fails to start, the log is at ${XDG_CACHE_HOME:-$HOME/.cache}/local-llm-sandbox/streamlit.log"
