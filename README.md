# Local-LLM-Sandbox

A sandbox to run LLMs locally and privately. A Streamlit web UI for your Ollama models with:

- **Model list** in the sidebar, pulled from whatever you have installed in Ollama, with install and delete built in
- **Voice input** (speech-to-text) using the mic in the chat bar, transcribed locally with faster-whisper
- **Voice replies** (text-to-speech) generated locally with Kokoro
- **Conversation memory** and an editable system prompt
- A dark three-panel layout: models on the left, chat in the middle, model info and voice settings on the right

Everything runs on your own machine. Prompts and recordings never leave it.

## Requirements

- Python **3.12** (Kokoro depends on `spacy`, which has no prebuilt packages for Python 3.13 or newer)
- [Ollama](https://ollama.com) with at least one model pulled
- `espeak-ng` (used by Kokoro for speech)
- A browser with microphone access

## Install on Fedora

```bash
# System packages
sudo dnf install espeak-ng python3.12 gcc gcc-c++

# Ollama and a model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Python packages (no virtual environment)
python3.12 -m ensurepip --user
python3.12 -m pip install --user -U pip
python3.12 -m pip install --user -r requirements.txt
python3.12 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
python3.12 -m pip install --user kokoro
```

The CPU-only PyTorch line keeps the download smaller. If you have an Nvidia GPU, skip it and run only the `kokoro` line.

### With a virtual environment instead

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install kokoro
```

## Install on Ubuntu / Debian

```bash
sudo apt install espeak-ng python3.12 python3.12-venv
```

Then follow the virtual environment steps above.

## Run

```bash
python3.12 -m streamlit run app.py --theme.base dark
```

Streamlit opens at `http://localhost:8501`. Allow microphone access when the browser asks. The first transcription and the first voice reply each download a model once, then everything works offline.

## Using it

- Click a model in the left sidebar to switch to it.
- Open **Manage models** in the sidebar to install or delete models:
  - **Install:** type a model name such as `llama3.2:3b` and click Install. A progress bar shows the download. Names are listed at [ollama.com/library](https://ollama.com/library). Keep the tab open until it finishes, since clicking elsewhere interrupts the download.
  - **Delete:** pick a model, tick the confirmation box, and click Delete.
- Type and press Enter to send, Shift + Enter for a new line.
- Click the mic in the chat bar to dictate. The transcription is sent automatically.
- Use the right panel to turn spoken replies on or off, pick a voice, replay the last reply, or edit the system prompt.
- **New chat** clears the conversation.

## Tuning

| What | Where in `app.py` | Change |
| --- | --- | --- |
| Transcription speed vs accuracy | `load_whisper()` | `"tiny"` is fastest, `"small"` is more accurate, `"base"` is the default |
| GPU transcription | `load_whisper()` | `device="cuda", compute_type="float16"` |
| Voices | `TTS_VOICES` | Kokoro voice names |
| British voices | `load_tts()` | `lang_code="b"` |

## Troubleshooting

**`Failed to build 'spacy'` when installing Kokoro**
You are on Python 3.13 or newer. Use Python 3.12 for every command (`python3.12 -m pip ...` and `python3.12 -m streamlit ...`).

**`TypeError: open() got an unexpected keyword argument 'metadata_errors'`**
faster-whisper and a newer PyAV (`av`) don't agree on a decoder option. The app avoids this by decoding the mic's WAV recording itself, so make sure you are running the latest `app.py`.

**The mic button isn't inside the chat bar**
That needs a recent Streamlit. Run `python3.12 -m pip install --user -U streamlit`. On older versions the app falls back to a small mic popover next to the chat bar.

**"Could not find Ollama"**
Make sure Ollama is running (`ollama serve`) and listening on `127.0.0.1:11434`.

**No voice reply**
Check that `espeak-ng` and `kokoro` are installed. The app shows a warning under the reply if Kokoro is missing.

**Changes not showing after an update**
Stop Streamlit with Ctrl+C and start it again. A browser refresh does not reload Python libraries.
