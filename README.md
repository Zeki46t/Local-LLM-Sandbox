# Local-LLM-Sandbox

A sandbox to run LLMs locally and privately. A Streamlit web UI for your Ollama models with:

- **Model list** in the sidebar, pulled from whatever you have installed in Ollama, with install and delete built in
- **Voice input** (speech-to-text) using the mic in the chat bar, transcribed locally with faster-whisper
- **Uploads** with the paperclip in the chat bar: documents, code, audio files and photos
- **Voice replies** (text-to-speech) generated locally with Kokoro
- **Saved chats** you can reopen or delete from the sidebar, plus an editable system prompt
- **One-click launch** from your app menu (see below)
- A dark three-panel layout: models and chats on the left, chat in the middle, model info and voice settings on the right

Everything runs on your own machine. Prompts and recordings never leave it.

## One-click app (quick start)

```bash
git clone https://github.com/Zeki46t/Local-LLM-Sandbox.git
cd Local-LLM-Sandbox
./install.sh --deps      # installs system packages and a private Python environment (.venv). Works on Pop!_OS, Ubuntu, Debian and Fedora
```

This adds **Local LLM Sandbox** to your app menu. Press the Super key, type "Local LLM", and click it. The launcher starts Ollama if it isn't running, starts the sandbox, and opens it in its own window (Chrome, Chromium, Brave or Edge) or your default browser. Clicking it again while it's running just reopens the window.

- **Stop it:** click **Stop app** at the bottom of the left sidebar. Closing the window leaves the server running in the background so reopening is instant.
- **Pin it:** while it's open, right-click its icon in the dock and choose Add to Favorites.
- **Remove it:** `./install.sh --uninstall` (your chats and models are kept).
- **Logs:** `~/.cache/local-llm-sandbox/streamlit.log` if something doesn't start.
- **Different port:** set `LLM_SANDBOX_PORT` (default 8501).
- Ollama itself still has to be installed from [ollama.com](https://ollama.com).

## Requirements

- Python **3.12** (Kokoro depends on `spacy`, which has no prebuilt packages for Python 3.13 or newer)
- [Ollama](https://ollama.com) with at least one model pulled
- `espeak-ng` (used by Kokoro for speech)
- `ffmpeg` (optional, only needed to upload audio other than WAV, such as mp3 or m4a)
- A browser with microphone access

## Install on Fedora

```bash
# System packages
sudo dnf install espeak-ng python3.12 gcc gcc-c++ ffmpeg-free

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

## Install on Ubuntu / Debian / Pop!_OS

The easiest way is `./install.sh --deps` (see the quick start). It uses a virtual environment because newer Ubuntu and Pop!_OS releases block system-wide `pip install` ("externally managed environment"). To do it by hand:

```bash
sudo apt install espeak-ng python3.12 python3.12-venv ffmpeg build-essential
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install kokoro
```

If your release has no `python3.12` package (Pop!_OS 22.04 and older), add the deadsnakes PPA first: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update`.

## Run

```bash
python3.12 -m streamlit run app.py --theme.base dark
```

(If you made the `.venv` above, use `.venv/bin/python -m streamlit run app.py --theme.base dark` instead.)

Streamlit opens at `http://localhost:8501`. Allow microphone access when the browser asks. The first transcription and the first voice reply each download a model once, then everything works offline.

## Using it

- Click a model in the left sidebar to switch to it.
- Open **Manage models** in the sidebar to install or delete models:
  - **Install:** type a model name such as `llama3.2:3b` and click Install. A progress bar shows the download. Names are listed at [ollama.com/library](https://ollama.com/library). Keep the tab open until it finishes, since clicking elsewhere interrupts the download.
  - **Delete:** pick a model, tick the confirmation box, and click Delete.
- Type and press Enter to send, Shift + Enter for a new line.
- Click the mic in the chat bar to dictate. The transcription is sent automatically.
- Click the paperclip in the chat bar to attach one or more files, then type a question (or just send them):
  - **Documents and code** (pdf, docx, txt, md, csv, json, py, js and more): the text is read and sent to the model along with your message.
  - **Audio** (wav, mp3, m4a, ogg, flac and more): transcribed locally with Whisper, and the transcript is sent to the model. Models can't hear audio directly. Anything other than WAV needs `ffmpeg`.
  - **Photos** (png, jpg, webp, gif): sent to the model as images. This only works with a vision model, such as `llama3.2-vision`, `gemma3` or `qwen2.5vl`. The right panel shows whether the active model supports images.
- Big files are trimmed to fit the **Context window** setting in the right panel. Raise it to fit more, at the cost of memory.
- Use the right panel to turn spoken replies on or off, pick a voice, replay the last reply, or edit the system prompt.
- **New chat** starts a fresh conversation.

## Saved chats

Chats are saved automatically after every reply and listed under **CHATS** in the left sidebar. Click one to reopen it (it switches back to the model you used, if it's still installed). Click ✕ to delete it.

- **Where:** `~/.local/share/local-llm-sandbox/` (one JSON file per chat in `chats/`, plus an `index.json`). The files are readable only by your user. Set `LLM_SANDBOX_DATA` to use a different folder.
- **What's saved:** your messages, the model's replies, the text pulled from attached files and transcripts, and any photos you attached. Spoken replies aren't saved.
- **Turn it off:** switch off **Save chats on this computer** under HISTORY in the right panel. New messages then aren't written to disk. Chats already saved stay until you delete them.
- **Delete everything:** remove the `~/.local/share/local-llm-sandbox/` folder.

## Tuning

| What | Where in `app.py` | Change |
| --- | --- | --- |
| Transcription speed vs accuracy | `load_whisper()` | `"tiny"` is fastest, `"small"` is more accurate, `"base"` is the default |
| GPU transcription | `load_whisper()` | `device="cuda", compute_type="float16"` |
| Voices | `TTS_VOICES` | Kokoro voice names |
| British voices | `load_tts()` | `lang_code="b"` |

## Performance and switching models

Language models are heavy, and switching between them in one chat is the most demanding thing the app does. To keep your computer responsive, the app:

- **Keeps one model in memory.** When you send a message with a different model selected, the previous model is unloaded first. A "Loading..." note shows while the new one starts, and the first reply after a switch is slower.
- **Warns about low RAM.** If the model you picked probably won't fit in the memory that's free right now, you get a warning before it loads instead of a frozen computer.
- **Leaves CPU cores free.** The **CPU threads** slider in the right panel defaults to all but two of your cores. Voice and transcription use about half.
- **Sends less to the model.** Older messages that don't fit the **Context window** are left out (a note tells you), and only the most recent photo is re-sent, because vision models re-process images on every turn.
- **Keeps the screen light.** Streaming replies refresh about 12 times a second instead of on every word, and the launcher turns off Streamlit's file watcher.
- **Reads long replies in part.** Spoken replies stop at a sentence end near 1,000 characters. The full text is still on screen.

If it still struggles, try a smaller model (3B or less, or a more compressed Q4 version), a smaller Context window, or fewer CPU threads. Switching models in a long chat is the slowest case, since the new model has to read the whole conversation again. **New chat** is faster.

To make Ollama itself hold only one model at a time, even outside this app:

```bash
sudo systemctl edit ollama
# add these two lines in the editor that opens, then save:
#   [Service]
#   Environment="OLLAMA_MAX_LOADED_MODELS=1"
#   Environment="OLLAMA_NUM_PARALLEL=1"
sudo systemctl restart ollama
```

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

**Audio upload fails with "only WAV audio can be read without ffmpeg"**
Install ffmpeg (`sudo dnf install ffmpeg-free` on Fedora, `sudo apt install ffmpeg` on Ubuntu), or upload a WAV file.

**"This model can't view images"**
Pull a vision model from **Manage models**, for example `llama3.2-vision`, and select it.

**PDF or Word upload says a package is missing**
Run `python3.12 -m pip install --user pypdf python-docx`.

**The app menu entry doesn't start anything**
Open `~/.cache/local-llm-sandbox/streamlit.log`. The most common cause is that the packages aren't installed for Python 3.12; run `./install.sh --deps`. You can also run `./launch.sh` in a terminal to see its messages.

**"Stop app" closed the window but I want it back**
Click Local LLM Sandbox in your app menu again. Your chats are saved, so reopen the one you want from the sidebar.

**"The Python packages aren't installed yet" (or the app menu entry does nothing)**
Run `./install.sh --deps` in the project folder. On Ubuntu and Pop!_OS this creates a `.venv` there, and `launch.sh` uses it automatically. If it says it can't install `python3.12`, add the deadsnakes PPA (see above).

**`error: externally-managed-environment` from pip**
Your system blocks system-wide installs. Use the `.venv` route (`./install.sh --deps`) instead of `pip install --user`.
