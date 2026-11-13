import base64
import html as htmllib
import inspect
import io
import re
import wave

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import ollama
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Local Sandbox", page_icon="🤖", layout="wide", initial_sidebar_state="expanded"
)

TTS_SAMPLE_RATE = 24000
TTS_VOICES = ["af_heart", "af_bella", "af_nicole", "bf_emma", "am_michael", "am_fenrir"]

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
    [data-testid="stSidebarUserContent"] {padding: 1.4rem 0.8rem;}
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
        position: fixed; top: 64px; right: 0; bottom: 0; width: 280px; z-index: 40;
        padding: 22px 24px; overflow-y: auto; background: var(--bg); border-left: 1px solid var(--line);
    }
    .mi-label {font-family: var(--mono); font-size: 0.64rem; letter-spacing: 0.14em; color: var(--faint); margin-bottom: 6px;}
    .mi-title {font-family: var(--serif); font-size: 1.5rem; color: var(--text); word-break: break-word;}
    .mi-sub {font-family: var(--serif); color: var(--muted); margin: 4px 0 14px;}
    .mi-row {
        display: flex; justify-content: space-between; gap: 10px; padding: 9px 0; border-top: 1px solid var(--line);
        font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.1em; color: var(--faint);
    }
    .mi-row b {color: var(--text); font-weight: 600; letter-spacing: 0.02em; text-align: right; word-break: break-all;}
    .mi-note {border-left: 2px solid var(--line2); padding-left: 12px; color: var(--muted); font-size: 0.85rem; margin: 14px 0;}
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
                    })
    except Exception:
        pass
    return infos


@st.cache_resource
def load_whisper():
    # "base" is quick; use "small" for better accuracy, "tiny" for max speed
    return WhisperModel("base", device="cpu", compute_type="int8")


def transcribe(audio_file):
    """Decode the WAV recording directly, so PyAV is never used."""
    model = load_whisper()
    with wave.open(io.BytesIO(audio_file.getvalue()), "rb") as wf:
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
    segments, _ = model.transcribe(data)
    return " ".join(seg.text.strip() for seg in segments).strip()


@st.cache_resource
def load_tts():
    # Imported here so the app still runs if Kokoro isn't installed
    from kokoro import KPipeline
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


def esc(text):
    return htmllib.escape(str(text))


def model_label(name):
    return esc(name.split(":")[0].upper())


def render_user(i, text):
    with st.container(key=f"urow-{i}"):
        st.markdown('<div class="lbl-you">YOU</div>', unsafe_allow_html=True)
        with st.container(key=f"ubub-{i}"):
            st.markdown(text)


def render_assistant(i, text, model):
    with st.container(key=f"arow-{i}"):
        st.markdown(f'<div class="lbl-ai">{model_label(model)}</div>', unsafe_allow_html=True)
        st.markdown(text)


def select_model(name):
    st.session_state.selected_model = name


# --- SESSION STATE ---
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
    with st.container(key="sidefoot"):
        st.markdown(
            '<div class="running"><i></i>Running locally</div>'
            '<div class="running-sub">Prompts and recordings stay on this computer.</div>',
            unsafe_allow_html=True,
        )

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
    rows_html = "".join(f'<div class="mi-row"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in rows)
    st.markdown(
        '<div class="mi-label">MODEL INFO</div>'
        f'<div class="mi-title">{esc(active.split(":")[0])}</div>'
        '<div class="mi-sub">Chat and code generation.</div>'
        f"{rows_html}",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="mi-label" style="margin-top:18px">VOICE</div>', unsafe_allow_html=True)
    tts_enabled = st.toggle("Speak replies", value=True)
    tts_voice = st.selectbox("Voice", options=TTS_VOICES, index=0, label_visibility="collapsed")
    if st.button("🔊 Replay last reply", disabled=not st.session_state.last_wav):
        play_hidden(st.session_state.last_wav)

    st.markdown('<div class="mi-label" style="margin-top:18px">SYSTEM PROMPT</div>', unsafe_allow_html=True)
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
            if st.button("↻ New chat"):
                st.session_state.messages = []
                st.session_state.last_wav = None
                st.rerun()
    st.markdown('<div class="page-rule"></div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown('<div class="empty">Ask anything.</div>', unsafe_allow_html=True)

    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            render_user(i, message["content"])
        else:
            render_assistant(i, message["content"], message.get("model", active))

# --- INPUT (mic lives inside the chat bar when Streamlit supports it) ---
prompt = None
supports_audio_input = "accept_audio" in inspect.signature(st.chat_input).parameters

if supports_audio_input:
    submission = st.chat_input("Ask anything...", accept_audio=True, audio_sample_rate=16000)
    if submission:
        if isinstance(submission, str):
            prompt = submission
        else:
            prompt = field(submission, "text")
            recording = field(submission, "audio")
            if not prompt and recording is not None:
                with st.spinner("Transcribing..."):
                    prompt = transcribe(recording)
                if not prompt:
                    st.warning("I couldn't hear anything in that recording.")
else:
    # Older Streamlit: normal chat bar plus a small mic popover
    # (run `pip install -U streamlit` to get the mic inside the chat bar)
    prompt = st.chat_input("Ask anything...")
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

# --- CORE LOGIC ---
if prompt:
    idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with center:
        render_user(idx, prompt)

        arow = st.container(key=f"arow-{idx + 1}")
        with arow:
            st.markdown(f'<div class="lbl-ai">{model_label(active)}</div>', unsafe_allow_html=True)
            response_placeholder = st.empty()

        full_response = ""
        try:
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            chat_messages = [{"role": "system", "content": system_prompt}] + history
            stream = ollama.chat(model=active, messages=chat_messages, stream=True)
            for chunk in stream:
                full_response += chunk['message']['content']
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "model": active}
            )
        except Exception as e:
            with arow:
                st.error(f"Error connecting to Ollama: {e}")

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
