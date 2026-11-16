import base64
import html as htmllib
import inspect
import io
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import ollama
from faster_whisper import WhisperModel

try:  # lets the background voice loader use Streamlit's cache without warnings
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except Exception:
    add_script_run_ctx = None

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Local Sandbox", page_icon="🤖", layout="wide", initial_sidebar_state="expanded"
)

TTS_SAMPLE_RATE = 24000
TTS_VOICES = ["af_heart", "af_bella", "af_nicole", "bf_emma", "am_michael", "am_fenrir"]

# Uploads: images go to vision models, audio is transcribed, everything else is read as text
IMAGE_EXTS = ["png", "jpg", "jpeg", "webp", "gif", "bmp"]
AUDIO_EXTS = ["mp3", "wav", "m4a", "ogg", "flac", "webm", "aac", "opus"]
DOC_EXTS = [
    "pdf", "docx", "txt", "md", "csv", "tsv", "json", "xml", "yaml", "yml", "toml", "ini", "log",
    "py", "js", "ts", "html", "css", "sql", "sh", "c", "cpp", "h", "java", "go", "rs",
]
UPLOAD_EXTS = IMAGE_EXTS + AUDIO_EXTS + DOC_EXTS
CONTEXT_SIZES = [4096, 8192, 16384, 32768]

# Performance: leave some cores free so the computer stays responsive while a model runs
CPU_CORES = max(2, os.cpu_count() or 4)
DEFAULT_THREADS = max(1, CPU_CORES - 2)
TTS_MAX_CHARS = 1000        # spoken replies are cut at a sentence end near this length
STREAM_UI_INTERVAL = 0.08   # seconds between on-screen updates while a reply streams

# Saved chats live here (set LLM_SANDBOX_DATA to put them somewhere else)
DATA_DIR = Path(os.environ.get("LLM_SANDBOX_DATA", Path.home() / ".local" / "share" / "local-llm-sandbox"))
CHAT_DIR = DATA_DIR / "chats"
INDEX_FILE = DATA_DIR / "index.json"
CHAT_ID_PATTERN = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")

# --- STYLING ---
st.markdown(
    """
    <style>
    :root {
        --bg: #0b0b0b; --panel: #0e0e0e; --raised: #161616; --line: #1f1f1f; --line2: #2a2a2a;
        --text: #ededed; --muted: #8a8a8a; --faint: #6b6b6b; --accent: #a78bfa; --ok: #4ade80;
        --serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Liberation Serif", "DejaVu Serif", serif;
        --mono: "JetBrains Mono", "DejaVu Sans Mono", "Liberation Mono", monospace;
    }

    /* Hide Streamlit chrome */
    /* (the toolbar itself stays, because the button that re-opens the sidebar can live inside it) */
    #MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"],
    [data-testid="stToolbarActions"], [data-testid="stMainMenu"], .stAppDeployButton {display: none !important;}
    [data-testid="stHeader"] {background: transparent;}

    /* Always keep the "open sidebar" button visible and on top of the top bar */
    [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {
        display: flex !important; visibility: visible !important; opacity: 1 !important;
        position: fixed; top: 14px; left: 14px; z-index: 1000001;
    }
    [data-testid="stExpandSidebarButton"] button, [data-testid="stSidebarCollapsedControl"] button {color: #d4d4d4;}
    /* When the sidebar is collapsed, the top bar spans the full width and leaves room for that button */
    .stApp:has([data-testid="stSidebar"][aria-expanded="false"]) .topbar {left: 0; padding-left: 64px;}

    /* When the sidebar closes, the chat stays centered and widens into the freed space (and eases back when it opens) */
    .topbar, .st-key-center, [data-testid="stBottomBlockContainer"] {
        transition: left 0.25s ease, padding 0.25s ease, max-width 0.25s ease;
    }
    .stApp:has([data-testid="stSidebar"][aria-expanded="false"]) .st-key-center,
    .stApp:has([data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stBottomBlockContainer"] {
        max-width: 1080px;
    }

    /* Base */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background: var(--bg); color: var(--text);}
    [data-testid="stMainBlockContainer"], .block-container {
        max-width: none; padding: 88px 280px 7rem 0;
    }

    /* Left sidebar */
    [data-testid="stSidebar"] {
        background: var(--panel); border-right: 1px solid var(--line);
        min-width: 256px; max-width: 256px;
    }
    [data-testid="stSidebarUserContent"] {padding: 1.4rem 0.8rem 120px;}
    .brand {font-family: var(--serif); font-size: 1.45rem; color: var(--text); line-height: 1.1; padding: 0 0.5rem;}
    .brand-sub {font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.14em; color: var(--faint); padding: 0.2rem 0.5rem 1.2rem;}
    .side-label {
        display: flex; justify-content: space-between; font-family: var(--mono);
        font-size: 0.64rem; letter-spacing: 0.14em; color: var(--faint); padding: 0.4rem 0.5rem;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%; justify-content: flex-start; text-align: left; height: auto;
        background: transparent; border: 1px solid transparent; border-radius: 8px;
        padding: 8px 12px; color: #d4d4d4; line-height: 1.4; position: relative;
    }
    [data-testid="stSidebar"] .stButton > button p {text-align: left; margin: 0; font-size: 0.92rem;}
    [data-testid="stSidebar"] .stButton > button code {
        background: transparent; color: var(--faint); padding: 0; font-family: var(--mono); font-size: 0.68rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {background: #131313; border-color: var(--line);}
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
        background: var(--raised); border-color: #232323; color: #fff;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] code,
    [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] code {color: var(--accent);}
    [data-testid="stSidebar"] .stButton > button[kind="primary"]::after,
    [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]::after {
        content: "●"; position: absolute; right: 12px; top: 50%; transform: translateY(-50%); font-size: 0.5rem; color: #ddd;
    }
    /* Manage models (install / delete) */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid var(--line); border-radius: 8px; background: transparent; margin-top: 0.8rem;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {font-size: 0.85rem; color: #d4d4d4;}
    [data-testid="stSidebar"] [data-testid="stForm"] {border: none; padding: 0;}
    [data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stFormSubmitButton"] > button {
        width: 100%; justify-content: center; text-align: center; background: transparent;
        border: 1px solid var(--line2); border-radius: 8px; color: #d4d4d4; padding: 6px 12px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button:hover,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stFormSubmitButton"] > button:hover {
        background: #151515; border-color: #3a3a3a; color: #fff;
    }
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #121212 !important; border-color: var(--line2) !important; color: var(--text) !important;
    }
    [class*="st-key-chatdel-"] button {
        justify-content: center !important; padding: 4px 0 !important; color: var(--faint) !important; min-height: 0;
    }
    [class*="st-key-chatdel-"] button:hover {color: #f87171 !important;}
    .st-key-stopapp button {
        background: transparent !important; border: none !important; color: var(--muted) !important;
        padding: 0 !important; margin-top: 6px; text-decoration: underline; font-size: 0.8rem; min-height: 0;
    }
    .st-key-stopapp button:hover {color: #f87171 !important;}
    .manage-note {font-size: 0.74rem; color: var(--faint); line-height: 1.4;}
    .manage-note a {color: var(--accent);}
    .st-key-sidefoot {position: fixed; bottom: 18px; left: 0; width: 256px; padding: 0 22px;}
    .running {font-size: 0.85rem; color: #d4d4d4;}
    .running i {display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--ok); margin-right: 8px;}
    .running-sub {font-size: 0.78rem; color: var(--faint); margin-top: 4px; line-height: 1.4;}

    /* Top bar */
    .topbar {
        position: fixed; top: 0; left: 256px; right: 0; height: 64px; z-index: 50;
        display: flex; align-items: center; justify-content: space-between; padding: 0 28px;
        background: var(--bg); border-bottom: 1px solid var(--line);
        font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.14em; color: var(--faint);
    }
    .topbar b {color: var(--text); font-weight: 600;}
    .topbar .status {letter-spacing: 0.02em; font-family: inherit; color: #cfcfcf; font-size: 0.78rem;}
    .topbar .status i {display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 8px; background: var(--ok);}
    .topbar .status.bad i {background: #f87171;}

    /* Center column */
    .st-key-center {max-width: 840px; margin: 0 auto; width: 100%;}
    .page-title {font-family: var(--serif); font-size: 2.6rem; line-height: 1.1; color: var(--text);}
    .page-sub {font-family: var(--serif); font-size: 1.05rem; color: var(--muted); margin: 0.5rem 0 1rem;}
    .page-rule {border-bottom: 1px solid var(--line); margin-bottom: 1.5rem;}
    .st-key-newchat {display: flex; justify-content: flex-end;}
    .st-key-newchat button {
        background: transparent; border: 1px solid var(--line2); border-radius: 8px; color: #d4d4d4; font-size: 0.85rem;
    }
    .st-key-newchat button:hover {background: #151515; border-color: #3a3a3a; color: #fff;}

    /* Messages */
    .lbl-you {text-align: right; font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.12em; color: var(--faint); margin-bottom: 2px;}
    .lbl-ai {font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.1em; color: var(--accent); margin: 0.6rem 0 2px;}
    [class*="st-key-ubub-"] {
        background: var(--raised); border: 1px solid #262626; border-radius: 10px;
        padding: 10px 16px; width: fit-content; max-width: 80%; margin-left: auto;
    }
    [class*="st-key-ubub-"] p {margin: 0; color: var(--text);}
    [class*="st-key-ubub-"] code {background: #222; color: #cfcfcf; font-size: 0.78rem; padding: 2px 7px; border-radius: 6px;}
    [class*="st-key-ubub-"] img {border-radius: 8px;}
    [class*="st-key-arow-"] [data-testid="stMarkdownContainer"] p,
    [class*="st-key-arow-"] [data-testid="stMarkdownContainer"] li {
        font-family: var(--serif); font-size: 1.12rem; line-height: 1.65; color: var(--text);
    }
    .empty {text-align: center; padding: 5rem 1rem; font-family: var(--serif); font-size: 1.2rem; color: var(--faint);}

    /* Chat input */
    [data-testid="stBottom"], [data-testid="stBottom"] > div {background: var(--bg);}
    [data-testid="stBottom"] {padding-right: 280px; box-sizing: border-box;}
    [data-testid="stBottomBlockContainer"] {max-width: 840px; margin: 0 auto; padding-bottom: 0.4rem;}
    [data-testid="stBottom"]::after {
        content: "Enter to send · Shift + Enter for a new line · Responses are generated locally and can make mistakes.";
        display: block; text-align: center; font-size: 0.72rem; color: var(--faint); padding: 0 0 14px;
    }
    [data-testid="stChatInput"] {background: #121212; border: 1px solid var(--line2); border-radius: 10px;}
    [data-testid="stChatInput"] textarea {color: var(--text);}
    [data-testid="stChatInputSubmitButton"] {background: var(--text); color: var(--bg); border-radius: 6px;}

    /* Right info panel */
    .st-key-infopanel {
        position: fixed; top: 64px; right: 0; bottom: 0; width: 280px; z-index: 1000; box-sizing: border-box;
        padding: 22px 24px 56px; overflow-y: auto; background: var(--bg); border-left: 1px solid var(--line);
    }
    .st-key-infopanel [data-testid="stVerticalBlock"] {gap: 0.55rem;}
    .st-key-infopanel [data-testid="stElementContainer"], .st-key-infopanel [data-testid="stMarkdownContainer"] {
        width: 100%; max-width: 100%;
    }
    .st-key-infopanel textarea {min-height: 120px; resize: none;}
    .mi-label {font-family: var(--mono); font-size: 0.64rem; letter-spacing: 0.14em; color: var(--faint); margin-bottom: 2px;}
    .mi-label.sec {margin-top: 1.1rem;}
    .mi-title {font-family: var(--serif); font-size: 1.5rem; color: var(--text); word-break: break-word;}
    .mi-sub {font-family: var(--serif); color: var(--muted); margin: 4px 0 14px;}
    .mi-row {
        display: flex; justify-content: space-between; gap: 10px; padding: 9px 0; border-top: 1px solid var(--line);
        font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.1em; color: var(--faint);
    }
    .mi-row b {color: var(--text); font-weight: 600; letter-spacing: 0.02em; text-align: right; word-break: break-all;}
    .mi-note {border-left: 2px solid var(--line2); padding-left: 12px; color: var(--muted); font-size: 0.85rem; margin: 6px 0 0;}
    .st-key-infopanel [data-baseweb="select"] > div, .st-key-infopanel textarea {
        background: #121212 !important; border-color: var(--line2) !important; color: var(--text) !important;
    }
    .st-key-infopanel .stButton > button {
        width: 100%; background: transparent; border: 1px solid var(--line2); border-radius: 8px; color: #d4d4d4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- FUNCTIONS ---
def field(obj, name):
    """Read a field from a dict or an object, whichever it is."""
    try:
        return obj[name]
    except Exception:
        return getattr(obj, name, None)


def get_model_infos():
    """Return a list of {name, params, quant, family} for installed Ollama models."""
    infos = []
    try:
        res = ollama.list()
        items = field(res, "models")
        if items is None and isinstance(res, list):
            items = res
        for m in items or []:
            name = field(m, "model") or field(m, "name")
            if not name:
                continue
            det = field(m, "details") or {}
            infos.append({
                "name": name,
                "params": field(det, "parameter_size") or "",
                "quant": field(det, "quantization_level") or "",
                "family": field(det, "family") or "",
                "size": field(m, "size") or 0,
            })
        if infos:
            return infos
    except Exception:
        pass
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                name = m.get("model") or m.get("name")
                det = m.get("details", {})
                if name:
                    infos.append({
                        "name": name,
                        "params": det.get("parameter_size", ""),
                        "quant": det.get("quantization_level", ""),
                        "family": det.get("family", ""),
                        "size": m.get("size", 0),
                    })
    except Exception:
        pass
    return infos


@st.cache_resource
def load_whisper():
    # "base" is quick; use "small" for better accuracy, "tiny" for max speed
    return WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=max(1, CPU_CORES // 2))


def wav_to_float(raw):
    """Decode WAV bytes to mono 16 kHz float32, so PyAV is never used for the mic."""
    with wave.open(io.BytesIO(raw), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise wave.Error("only 16-bit WAV is read directly")
        rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        n = int(len(data) * 16000 / rate)
        data = np.interp(
            np.linspace(0, len(data) - 1, n), np.arange(len(data)), data
        ).astype(np.float32)
    return data


def decode_audio_bytes(raw):
    """WAV is read directly; mp3, m4a, ogg and the rest go through ffmpeg if it's installed."""
    try:
        return wav_to_float(raw)
    except (wave.Error, EOFError):
        pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "only WAV audio can be read without ffmpeg. Install it (Fedora: sudo dnf install ffmpeg-free) "
            "or upload a WAV file."
        )
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=raw, capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("ffmpeg couldn't read that audio file: " + proc.stderr.decode(errors="replace")[:200])
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_bytes(raw):
    model = load_whisper()
    segments, _ = model.transcribe(decode_audio_bytes(raw))
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe(audio_file):
    return transcribe_bytes(audio_file.getvalue())


def file_ext(name):
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def prepare_image(raw, max_side=1280):
    """Shrink big photos so vision models answer faster. Falls back to the original bytes."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        return raw


def extract_text(name, raw):
    """Pull readable text out of a PDF, Word file or plain text/code file."""
    ext = file_ext(name)
    if ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("PDF support needs pypdf (pip install pypdf)")
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if ext == "docx":
        try:
            import docx
        except ImportError:
            raise RuntimeError("Word support needs python-docx (pip install python-docx)")
        document = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in document.paragraphs).strip()
    return raw.decode("utf-8", errors="replace").strip()


def shorten(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated to fit the context window]"


def process_uploads(files, num_ctx):
    """Turn uploaded files into (attachments, images, extra_text, errors) for the model."""
    attachments, images, parts, errors = [], [], [], []
    text_files = [f for f in files if file_ext(f.name) not in IMAGE_EXTS]
    # Roughly 2.5 characters per token, spread across the text-based files
    # Keep attachments to ~60% of the window so there's room for the chat and the reply
    per_file = max(2000, int(num_ctx * 2.5 * 0.6) // max(1, len(text_files)))
    for f in files:
        raw = f.getvalue()
        ext = file_ext(f.name)
        try:
            if ext in IMAGE_EXTS:
                images.append(prepare_image(raw))
                attachments.append({"name": f.name, "kind": "image"})
            elif ext in AUDIO_EXTS:
                text = transcribe_bytes(raw)
                if not text:
                    errors.append(f"{f.name}: no speech found")
                    continue
                parts.append(f'[Transcript of audio file "{f.name}"]\n{shorten(text, per_file)}')
                attachments.append({"name": f.name, "kind": "audio"})
            else:
                text = extract_text(f.name, raw)
                if not text:
                    errors.append(f"{f.name}: no readable text found")
                    continue
                parts.append(f'[Attached file "{f.name}"]\n```\n{shorten(text, per_file)}\n```')
                attachments.append({"name": f.name, "kind": "file"})
        except Exception as e:
            errors.append(f"{f.name}: {e}")
    return attachments, images, "\n\n".join(parts), errors


@st.cache_data(ttl=60, show_spinner=False)
def has_vision(model_name):
    """True or False if Ollama reports it, None if it can't tell."""
    try:
        caps = field(ollama.show(model_name), "capabilities")
        if caps:
            return "vision" in caps
    except Exception:
        pass
    return None


@st.cache_resource
def load_tts():
    # Imported here so the app still runs if Kokoro isn't installed
    import torch
    from kokoro import KPipeline
    torch.set_num_threads(max(1, CPU_CORES // 2))  # leave cores for the language model and the system
    return KPipeline(lang_code="a")  # "a" = American English; use "b" for British voices


def clean_for_speech(text):
    """Strip markdown so the voice doesn't read out symbols."""
    text = re.sub(r"```.*?```", " Code block omitted. ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_#>~|]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def synthesize(text, voice):
    """Return WAV bytes for the given text, generated locally with Kokoro."""
    text = clean_for_speech(text)
    if not text:
        return None
    if len(text) > TTS_MAX_CHARS:
        # Speaking a very long reply keeps the CPU pegged, so read the opening and stop at a sentence end
        cut = text[:TTS_MAX_CHARS]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        text = cut[: end + 1] if end > 200 else cut
    pipeline = load_tts()
    chunks = [np.asarray(audio, dtype=np.float32) for _, _, audio in pipeline(text, voice=voice)]
    if not chunks:
        return None
    pcm = (np.clip(np.concatenate(chunks), -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def play_hidden(wav_bytes):
    """Autoplay audio with no visible player."""
    b64 = base64.b64encode(wav_bytes).decode()
    components.html(
        f'<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>',
        height=0,
    )


# --- VOICE PRELOAD START ---
def _warm_up_voice(state):
    """Load speech-to-text and text-to-speech ahead of time, then run each once so the first real use is fast."""
    try:
        model = load_whisper()
        list(model.transcribe(np.zeros(16000, dtype=np.float32))[0])  # one second of silence
        state["stt"] = True
    except Exception as e:
        state["stt_error"] = str(e)
    try:
        pipeline = load_tts()
        for voice in TTS_VOICES:  # fetch and keep every voice ready, so switching voices is instant too
            try:
                pipeline.load_voice(voice)
            except Exception:
                pass
        for _ in pipeline("Ready.", voice=TTS_VOICES[0]):
            pass
        state["tts"] = True
    except Exception as e:
        state["tts_error"] = str(e)
    state["done"] = True


@st.cache_resource
def start_voice_preload():
    """Start loading the voice models in the background, once per app launch.
    Set LLM_SANDBOX_PRELOAD=0 to skip this (saves about 1 GB of RAM until voice is first used)."""
    state = {"done": False}
    if os.environ.get("LLM_SANDBOX_PRELOAD", "1") == "0":
        state["done"] = True
        return state
    thread = threading.Thread(target=_warm_up_voice, args=(state,), daemon=True, name="voice-preload")
    if add_script_run_ctx:
        try:
            add_script_run_ctx(thread)
        except Exception:
            pass
    thread.start()
    return state
# --- VOICE PRELOAD END ---


def esc(text):
    return htmllib.escape(str(text))


def model_label(name):
    return esc(name.split(":")[0].upper())


ATTACH_ICONS = {"image": "🖼️", "audio": "🎧", "file": "📄"}


def render_user(i, text, attachments=None, images=None):
    with st.container(key=f"urow-{i}"):
        st.markdown('<div class="lbl-you">YOU</div>', unsafe_allow_html=True)
        with st.container(key=f"ubub-{i}"):
            if text:
                st.markdown(text)
            if attachments:
                chips = "  ".join(f"`{ATTACH_ICONS.get(a['kind'], '📎')} {a['name']}`" for a in attachments)
                st.markdown(chips)
            for img in images or []:
                st.image(img, width=240)


def render_assistant(i, text, model):
    with st.container(key=f"arow-{i}"):
        st.markdown(f'<div class="lbl-ai">{model_label(model)}</div>', unsafe_allow_html=True)
        st.markdown(text)


def select_model(name):
    st.session_state.selected_model = name


def normalize_model_name(name):
    """Ollama lists untagged models as ':latest', so match that."""
    name = name.strip()
    return name if ":" in name else f"{name}:latest"


def pull_model(name, bar, status):
    """Download a model from Ollama, updating a progress bar as it goes."""
    for p in ollama.pull(name, stream=True):
        text = field(p, "status") or ""
        total = field(p, "total")
        done = field(p, "completed")
        if total and done is not None:
            bar.progress(min(done / total, 1.0), text=text)
        else:
            status.caption(text)


def delete_model(name):
    """Runs as a button callback, so it can also reset the confirm checkbox."""
    try:
        ollama.delete(name)
        st.session_state.flash = f"Deleted {name}"
    except Exception as e:
        st.session_state.flash = f"Couldn't delete {name}: {e}"
    st.session_state.delete_confirm = False


# --- PERFORMANCE HELPERS START ---
def unload_other_models(active_name):
    """Keep only the active model in memory.
    Returns (names unloaded, whether the active model was already loaded)."""
    unloaded, was_loaded = [], False
    try:
        for m in field(ollama.ps(), "models") or []:
            name = field(m, "model") or field(m, "name")
            if not name:
                continue
            if name == active_name:
                was_loaded = True
            else:
                try:
                    ollama.generate(model=name, prompt="", keep_alive=0)
                    unloaded.append(name)
                except Exception:
                    pass
    except Exception:
        pass
    return unloaded, was_loaded


def free_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return None


def ram_warning(model_name, size_bytes):
    """A heads-up if the model probably won't fit in the RAM that's free right now."""
    if not size_bytes:
        return None
    free = free_ram_gb()
    need = size_bytes / 1e9 * 1.2 + 0.5
    if free is not None and free < need:
        return (
            f"Only {free:.1f} GB of RAM is free and {model_name} needs roughly {need:.1f} GB. "
            "Your computer may slow down or freeze while it loads. Close other apps or pick a smaller model."
        )
    return None


def build_history(messages, vision_ok, num_ctx):
    """Chat history for the model: newest messages that fit the window, and photos only from the latest
    message that has them. Returns (history, number of older messages left out)."""
    last_img = max((i for i, m in enumerate(messages) if m.get("images")), default=-1)
    entries = []
    for i, m in enumerate(messages):
        e = {"role": m["role"], "content": m.get("model_content", m["content"])}
        if vision_ok and i == last_img:
            e["images"] = m["images"]
        entries.append(e)
    budget = int(num_ctx * 2.5 * 0.65)  # leave room for the system prompt and the reply
    kept, used = [], 0
    for e in reversed(entries):
        size = len(e["content"])
        if kept and used + size > budget:
            break
        kept.append(e)
        used += size
    kept.reverse()
    while len(kept) > 1 and kept[0]["role"] != "user":
        kept.pop(0)
    return kept, len(entries) - len(kept)
# --- PERFORMANCE HELPERS END ---


# --- SAVED CHATS (HISTORY START) ---
def _write_json(path, data):
    """Write atomically and keep the files private to your user."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(DATA_DIR, 0o700)
    except OSError:
        pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def messages_to_json(messages):
    out = []
    for m in messages:
        item = {k: v for k, v in m.items() if k != "images"}
        if m.get("images"):
            item["images"] = [base64.b64encode(b).decode() for b in m["images"]]
        out.append(item)
    return out


def messages_from_json(items):
    out = []
    for m in items:
        m = dict(m)
        if m.get("images"):
            m["images"] = [base64.b64decode(s) for s in m["images"]]
        out.append(m)
    return out


def rebuild_index():
    index = {}
    for p in CHAT_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            index[d["id"]] = {
                "title": d.get("title", "Untitled"),
                "updated": d.get("updated", ""),
                "count": len(d.get("messages", [])),
                "model": d.get("model", ""),
            }
        except Exception:
            continue
    if index:
        try:
            _write_json(INDEX_FILE, index)
        except Exception:
            pass
    return index


def load_index():
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return rebuild_index()


def make_title(messages):
    for m in messages:
        if m["role"] == "user":
            t = (m.get("content") or "").strip()
            if not t and m.get("attachments"):
                t = m["attachments"][0]["name"]
            t = " ".join(t.split())
            if t:
                return t[:48] + ("…" if len(t) > 48 else "")
    return "New chat"


def save_chat():
    """Save the current conversation to disk, if saving is switched on."""
    if not st.session_state.get("save_history", True) or not st.session_state.messages:
        return
    try:
        if not st.session_state.chat_id:
            st.session_state.chat_id = (
                datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            )
        cid = st.session_state.chat_id
        now = datetime.now().isoformat(timespec="seconds")
        index = load_index()
        title = index.get(cid, {}).get("title") or make_title(st.session_state.messages)
        data = {
            "id": cid,
            "title": title,
            "updated": now,
            "model": st.session_state.selected_model,
            "messages": messages_to_json(st.session_state.messages),
        }
        _write_json(CHAT_DIR / f"{cid}.json", data)
        index[cid] = {
            "title": title,
            "updated": now,
            "count": len(st.session_state.messages),
            "model": st.session_state.selected_model,
        }
        _write_json(INDEX_FILE, index)
    except Exception as e:
        st.session_state.flash = f"Couldn't save chat: {e}"


def new_chat():
    st.session_state.messages = []
    st.session_state.chat_id = None
    st.session_state.last_wav = None


def open_chat(cid):
    if not CHAT_ID_PATTERN.fullmatch(cid):
        return
    try:
        data = json.loads((CHAT_DIR / f"{cid}.json").read_text(encoding="utf-8"))
        st.session_state.messages = messages_from_json(data["messages"])
        st.session_state.chat_id = cid
        st.session_state.last_wav = None
        if data.get("model"):
            st.session_state.selected_model = data["model"]  # falls back to the first model if it's gone
    except Exception as e:
        st.session_state.flash = f"Couldn't open that chat: {e}"


def delete_chat(cid):
    if not CHAT_ID_PATTERN.fullmatch(cid):
        return
    try:
        (CHAT_DIR / f"{cid}.json").unlink(missing_ok=True)
        index = load_index()
        index.pop(cid, None)
        _write_json(INDEX_FILE, index)
    except Exception as e:
        st.session_state.flash = f"Couldn't delete that chat: {e}"
    if st.session_state.get("chat_id") == cid:
        new_chat()


def md_escape(text):
    return re.sub(r"([\\`*_{}\[\]<>#|~$])", r"\\\1", text)


def format_when(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %H:%M")
    except Exception:
        return ""


def render_chat_list(slot, tag):
    """Fill the sidebar's chat list. `tag` keeps widget keys unique if it's drawn twice in one run."""
    index = load_index()
    items = sorted(index.items(), key=lambda kv: kv[1].get("updated", ""), reverse=True)[:25]
    with slot.container():
        st.markdown(
            f'<div class="side-label"><span>CHATS</span><span>{len(index):02d}</span></div>',
            unsafe_allow_html=True,
        )
        if not items:
            st.caption("Your saved chats will show up here.")
        for cid, meta in items:
            c1, c2 = st.columns([6, 1], vertical_alignment="center")
            with c1:
                st.button(
                    f"{md_escape(meta.get('title', 'Untitled'))}  \n`{format_when(meta.get('updated', ''))}`",
                    key=f"chat-{cid}-{tag}",
                    type="primary" if cid == st.session_state.chat_id else "secondary",
                    on_click=open_chat,
                    args=(cid,),
                    use_container_width=True,
                )
            with c2:
                st.button("✕", key=f"chatdel-{cid}-{tag}", on_click=delete_chat, args=(cid,), help="Delete this chat")
# --- SAVED CHATS (HISTORY END) ---


# Start loading the voice models in the background so the first spoken reply isn't slow
start_voice_preload()

# --- SESSION STATE ---
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
if "last_wav" not in st.session_state:
    st.session_state.last_wav = None

infos = get_model_infos()
names = [m["name"] for m in infos]
if names and st.session_state.selected_model not in names:
    st.session_state.selected_model = names[0]
active = st.session_state.selected_model or "llama3"
active_info = next((m for m in infos if m["name"] == active), {})

# --- LEFT SIDEBAR: BRAND + MODEL LIST ---
flash = st.session_state.pop("flash", None)
if flash:
    st.toast(flash)

with st.sidebar:
    st.markdown(
        '<div class="brand">Local Sandbox</div><div class="brand-sub">LOCAL INFERENCE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="side-label"><span>MODELS</span><span>{len(infos):02d}</span></div>',
        unsafe_allow_html=True,
    )
    if not infos:
        st.error("Could not find Ollama.")
    for m in infos:
        meta = " · ".join(x for x in [m["params"], m["quant"]] if x) or "local model"
        st.button(
            f"{m['name']}  \n`{meta}`",
            key=f"model-{m['name']}",
            type="primary" if m["name"] == active else "secondary",
            on_click=select_model,
            args=(m["name"],),
            use_container_width=True,
        )
    with st.expander("➕ Manage models"):
        st.markdown('<div class="side-label" style="padding:0 0 0.2rem">INSTALL</div>', unsafe_allow_html=True)
        with st.form("pull_form", clear_on_submit=True):
            new_model = st.text_input(
                "Model name", placeholder="e.g. llama3.2:3b", label_visibility="collapsed"
            )
            pull_clicked = st.form_submit_button("Install model")
        st.markdown(
            '<div class="manage-note">Browse names at '
            '<a href="https://ollama.com/library" target="_blank">ollama.com/library</a>. '
            "Downloads can be several GB, so keep this tab open until it finishes.</div>",
            unsafe_allow_html=True,
        )
        pulled = None
        if pull_clicked and new_model.strip():
            target = normalize_model_name(new_model)
            bar = st.progress(0.0, text=f"Starting {target}...")
            status = st.empty()
            try:
                pull_model(target, bar, status)
                pulled = target
            except Exception as e:
                bar.empty()
                st.error(f"Couldn't install {target}: {e}")
        if pulled:
            st.session_state.selected_model = pulled
            st.session_state.flash = f"Installed {pulled}"
            st.rerun()

        st.markdown(
            '<div class="side-label" style="padding:0.8rem 0 0.2rem">DELETE</div>', unsafe_allow_html=True
        )
        if names:
            to_delete = st.selectbox(
                "Model to delete", options=names, label_visibility="collapsed", key="delete_choice"
            )
            confirm = st.checkbox(f"Yes, delete {to_delete}", key="delete_confirm")
            st.button(
                "Delete model",
                disabled=not confirm,
                on_click=delete_model,
                args=(to_delete,),
                key="delete_button",
            )
        else:
            st.caption("No models installed.")

    chat_slot = st.empty()  # the saved-chat list is drawn here (see render_chat_list)

    with st.container(key="sidefoot"):
        st.markdown(
            '<div class="running"><i></i>Running locally</div>'
            '<div class="running-sub">Prompts and recordings stay on this computer.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Stop app", key="stopapp", help="Shuts down the sandbox. Open it again from your app menu."):
            os.kill(os.getpid(), signal.SIGTERM)

# --- TOP BAR ---
status_class = "" if infos else "bad"
status_text = "Model ready" if infos else "Ollama not found"
st.markdown(
    f'<div class="topbar"><div>SANDBOX &nbsp;/&nbsp; <b>CHAT</b></div>'
    f'<div class="status {status_class}"><i></i>{status_text}</div></div>',
    unsafe_allow_html=True,
)

# --- RIGHT INFO PANEL ---
with st.container(key="infopanel"):
    rows = [("MODEL", active)]
    if active_info.get("params"):
        rows.append(("PARAMETERS", active_info["params"]))
    if active_info.get("quant"):
        rows.append(("FORMAT", active_info["quant"]))
    if active_info.get("family"):
        rows.append(("FAMILY", active_info["family"]))
    vision = has_vision(active)
    if vision is not None:
        rows.append(("IMAGES", "supported" if vision else "not supported"))
    rows_html = "".join(f'<div class="mi-row"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in rows)
    st.markdown(
        '<div class="mi-label">MODEL INFO</div>'
        f'<div class="mi-title">{esc(active.split(":")[0])}</div>'
        '<div class="mi-sub">Chat and code generation.</div>'
        f"{rows_html}",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="mi-label sec">VOICE</div>', unsafe_allow_html=True)
    tts_enabled = st.toggle("Speak replies", value=True)
    tts_voice = st.selectbox("Voice", options=TTS_VOICES, index=0, label_visibility="collapsed")
    if st.button("🔊 Replay last reply", disabled=not st.session_state.last_wav):
        play_hidden(st.session_state.last_wav)

    st.markdown('<div class="mi-label sec">CONTEXT WINDOW</div>', unsafe_allow_html=True)
    num_ctx = st.selectbox(
        "Context window",
        options=CONTEXT_SIZES,
        index=1,
        format_func=lambda n: f"{n:,} tokens",
        label_visibility="collapsed",
        help="How much text the model can see at once. Larger values fit bigger attached files but use more memory.",
    )

    st.markdown('<div class="mi-label sec">CPU THREADS</div>', unsafe_allow_html=True)
    num_threads = st.slider(
        "CPU threads",
        1,
        CPU_CORES,
        DEFAULT_THREADS,
        label_visibility="collapsed",
        help="How many CPU cores the model may use. Leaving a couple free keeps the rest of your computer responsive.",
    )

    st.markdown('<div class="mi-label sec">HISTORY</div>', unsafe_allow_html=True)
    st.toggle(
        "Save chats on this computer",
        value=True,
        key="save_history",
        help=f"Chats are stored in {DATA_DIR}. Turn this off to stop saving new messages.",
    )

    st.markdown('<div class="mi-label sec">SYSTEM PROMPT</div>', unsafe_allow_html=True)
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful AI assistant running in a local sandbox.",
        height=120,
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="mi-note">Generated code blocks have a copy button.</div>',
        unsafe_allow_html=True,
    )

# --- CENTER: HEADER + HISTORY ---
center = st.container(key="center")
with center:
    head_l, head_r = st.columns([5, 1.3], vertical_alignment="bottom")
    with head_l:
        st.markdown(
            f'<div class="page-title">Chat</div>'
            f'<div class="page-sub">Chat and generate code with {esc(active)}.</div>',
            unsafe_allow_html=True,
        )
    with head_r:
        with st.container(key="newchat"):
            st.button("↻ New chat", on_click=new_chat)
    st.markdown('<div class="page-rule"></div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown('<div class="empty">Ask anything.</div>', unsafe_allow_html=True)

    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            render_user(i, message["content"], message.get("attachments"), message.get("images"))
        else:
            render_assistant(i, message["content"], message.get("model", active))

# --- INPUT (mic and file uploads live inside the chat bar when Streamlit supports it) ---
prompt = None
uploads = []
chat_params = inspect.signature(st.chat_input).parameters
supports_audio_input = "accept_audio" in chat_params
supports_file_input = "accept_file" in chat_params

chat_kwargs = {}
if supports_audio_input:
    chat_kwargs.update(accept_audio=True, audio_sample_rate=16000)
if supports_file_input:
    chat_kwargs.update(accept_file="multiple", file_type=UPLOAD_EXTS)

if chat_kwargs:
    submission = st.chat_input("Ask anything...", **chat_kwargs)
    if submission:
        if isinstance(submission, str):
            prompt = submission
        else:
            prompt = field(submission, "text")
            uploads = list(field(submission, "files") or [])
            recording = field(submission, "audio")
            if not prompt and recording is not None:
                with st.spinner("Transcribing..."):
                    prompt = transcribe(recording)
                if not prompt:
                    st.warning("I couldn't hear anything in that recording.")
else:
    prompt = st.chat_input("Ask anything...")

# Older Streamlit: small popovers next to the chat bar
# (run `pip install -U streamlit` to get the mic and paperclip inside the chat bar)
if not supports_file_input:
    if "uploader_n" not in st.session_state:
        st.session_state.uploader_n = 0
    with st.popover("📎"):
        fallback_files = st.file_uploader(
            "Attach files, photos or audio",
            type=UPLOAD_EXTS,
            accept_multiple_files=True,
            key=f"uploader-{st.session_state.uploader_n}",
        )
    uploads = list(fallback_files or [])

if not supports_audio_input:
    with st.popover("🎤"):
        recording = st.audio_input("Record a message", label_visibility="collapsed")
    if recording is not None:
        audio_id = hash(recording.getvalue())
        if audio_id != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio_id
            with st.spinner("Transcribing..."):
                spoken_text = transcribe(recording)
            if spoken_text:
                prompt = spoken_text
            else:
                st.warning("I couldn't hear anything in that recording.")

# --- SAVED CHAT LIST (sidebar) ---
render_chat_list(chat_slot, "a")

# --- CORE LOGIC ---
if prompt or uploads:
    idx = len(st.session_state.messages)
    text = prompt or ""
    attachments, images, extra_text, upload_errors = [], [], "", []
    if uploads:
        with center, st.spinner("Reading attachments..."):
            attachments, images, extra_text, upload_errors = process_uploads(uploads, num_ctx)
        if not supports_file_input:
            st.session_state.uploader_n += 1  # clears the fallback uploader after sending

    # Nothing usable (every attachment failed and no text): show why and stop
    if not text and not attachments and not images:
        with center:
            for err in upload_errors:
                st.warning(f"Couldn't read {err}")
        st.stop()

    model_text = text or "Please take a look at what I attached."
    if extra_text:
        model_text = f"{model_text}\n\n{extra_text}"
    vision_ok = has_vision(active) is not False
    if images and not vision_ok:
        model_text += f"\n\n[{len(images)} image(s) were attached, but this model can't view images.]"

    st.session_state.messages.append({
        "role": "user",
        "content": text,
        "model_content": model_text,
        "attachments": attachments,
        "images": images,
    })

    with center:
        render_user(idx, text, attachments, images)
        for err in upload_errors:
            st.warning(f"Couldn't read {err}")
        if images and not vision_ok:
            st.warning(
                f"{active} can't view images, so the photo was not sent. "
                "Pick a vision model (for example llama3.2-vision, gemma3 or qwen2.5vl) from Manage models."
            )

        arow = st.container(key=f"arow-{idx + 1}")
        with arow:
            st.markdown(f'<div class="lbl-ai">{model_label(active)}</div>', unsafe_allow_html=True)
            note_slot = st.empty()
            response_placeholder = st.empty()

        full_response = ""
        try:
            # Keep only one model in memory: two at once is what exhausts RAM and freezes the system
            unloaded, was_loaded = unload_other_models(active)
            if unloaded:
                st.toast(f"Unloaded {', '.join(unloaded)} to free memory")
            if not was_loaded:
                response_placeholder.caption(
                    f"Loading {active}... the first reply after switching models is slower."
                )

            history, dropped = build_history(st.session_state.messages, vision_ok, num_ctx)
            notes = []
            if not was_loaded:
                warn = ram_warning(active, active_info.get("size", 0))
                if warn:
                    notes.append(("warning", warn))
            if dropped:
                notes.append(("caption", f"{dropped} older message(s) were left out to fit the context window."))
            if notes:
                with note_slot.container():
                    for kind, msg in notes:
                        getattr(st, kind)(msg)

            chat_messages = [{"role": "system", "content": system_prompt}] + history
            stream = ollama.chat(
                model=active,
                messages=chat_messages,
                stream=True,
                options={"num_ctx": num_ctx, "num_thread": num_threads},
            )
            last_ui = 0.0
            for chunk in stream:
                full_response += chunk['message']['content']
                now = time.monotonic()
                if now - last_ui >= STREAM_UI_INTERVAL:
                    response_placeholder.markdown(full_response + "▌")
                    last_ui = now
            response_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "model": active}
            )
        except Exception as e:
            with arow:
                st.error(f"Error connecting to Ollama: {e}")
                if images:
                    st.caption("If this model can't handle images, pick a vision model such as llama3.2-vision or gemma3.")

        # Save the conversation and refresh the sidebar list so the new chat shows up
        save_chat()
        render_chat_list(chat_slot, "b")

        # Speak the reply with no visible player
        if tts_enabled and full_response:
            with arow:
                try:
                    with st.spinner("Generating voice..."):
                        wav_bytes = synthesize(full_response, tts_voice)
                    if wav_bytes:
                        st.session_state.last_wav = wav_bytes
                        play_hidden(wav_bytes)
                except ImportError:
                    st.warning("Voice is off because Kokoro isn't installed.")
                except Exception as e:
                    st.warning(f"Couldn't generate voice: {e}")
