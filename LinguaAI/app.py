"""
LinguaAI – Intelligent Multilingual Translation Platform
=========================================================
Enterprise-grade AI-powered translation platform built with Streamlit.
Supports 100+ languages, NLP analytics, TTS, translation history.

Author  : Sai Sathwik
Stack   : Python 3.13 · Streamlit · deep-translator · gTTS · LangDetect · NLTK
"""

import io
import os
import sys
import time
import json
import base64
import datetime
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils.translator import (
    translate_text,
    LANGUAGES,
    SOURCE_LANGUAGES,
    TARGET_LANGUAGES,
    CODE_TO_NAME,
)
from utils.language_detector import detect_language, analyze_text
from utils.tts import text_to_speech, get_tts_languages

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def bootstrap_nltk():
    try:
        import nltk
        for pkg in ["punkt", "punkt_tab", "stopwords"]:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    except Exception:
        pass

bootstrap_nltk()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinguaAI – Multilingual Translation Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "LinguaAI – Intelligent Multilingual Translation Platform"},
)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS / DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design Tokens ── */
:root {
  --cyan:    #00D4FF;
  --violet:  #6C63FF;
  --mint:    #00FFB2;
  --rose:    #FF4D6D;
  --amber:   #FFB703;
  --bg0:     #020812;
  --bg1:     #070F1E;
  --bg2:     #0D1929;
  --bg3:     #112236;
  --surface: rgba(13, 25, 41, 0.85);
  --glass:   rgba(255,255,255,0.04);
  --border:  rgba(0, 212, 255, 0.15);
  --text-hi: #F0F8FF;
  --text-md: #8BA7C4;
  --text-lo: #3D5970;
  --radius:  14px;
  --shadow:  0 8px 32px rgba(0,0,0,0.6);
}

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg0) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--text-hi);
}

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,212,255,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 110%, rgba(108,99,255,0.08) 0%, transparent 60%);
  pointer-events: none;
  z-index: 0;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stSidebar"] { background: var(--bg1) !important; }

/* ── Remove Streamlit padding ── */
.main .block-container {
  padding: 0 2rem 4rem !important;
  max-width: 1400px;
  margin: 0 auto;
}

/* ── Hero Banner ── */
.hero {
  position: relative;
  width: 100%;
  min-height: 420px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 3rem 1rem 2rem;
  overflow: hidden;
  margin-bottom: 2rem;
}

.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    linear-gradient(135deg, rgba(0,212,255,0.06) 0%, transparent 50%),
    linear-gradient(225deg, rgba(108,99,255,0.06) 0%, transparent 50%);
  border-bottom: 1px solid var(--border);
}

.hero-globe {
  font-size: 4rem;
  filter: drop-shadow(0 0 30px rgba(0,212,255,0.6));
  margin-bottom: 1rem;
  animation: floatGlobe 4s ease-in-out infinite;
}

@keyframes floatGlobe {
  0%, 100% { transform: translateY(0px); }
  50%       { transform: translateY(-12px); }
}

.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(2rem, 5vw, 3.8rem);
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.1;
  background: linear-gradient(135deg, var(--cyan) 0%, var(--violet) 50%, var(--mint) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.75rem;
}

.hero-sub {
  font-size: 1.1rem;
  color: var(--text-md);
  font-weight: 300;
  letter-spacing: 0.02em;
  max-width: 600px;
  margin: 0 auto 1.5rem;
}

.hero-badges {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  justify-content: center;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.35rem 0.9rem;
  border-radius: 50px;
  border: 1px solid var(--border);
  background: var(--glass);
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--text-md);
  backdrop-filter: blur(8px);
  letter-spacing: 0.03em;
}

.badge.active { color: var(--cyan); border-color: rgba(0,212,255,0.4); }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--bg2) !important;
  border-radius: 12px !important;
  padding: 4px !important;
  border: 1px solid var(--border) !important;
  gap: 4px !important;
  margin-bottom: 1.5rem !important;
}

[data-testid="stTabs"] [role="tab"] {
  border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.85rem !important;
  color: var(--text-md) !important;
  border: none !important;
  padding: 0.5rem 1.2rem !important;
  transition: all 0.2s ease !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(108,99,255,0.15)) !important;
  color: var(--cyan) !important;
  box-shadow: 0 0 0 1px rgba(0,212,255,0.3) !important;
}

/* ── Glass Cards ── */
.glass-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  backdrop-filter: blur(16px);
  box-shadow: var(--shadow);
  margin-bottom: 1rem;
}

.glass-card-sm {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  backdrop-filter: blur(12px);
}

/* ── KPI Cards ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.kpi-card {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem 1rem;
  text-align: center;
  backdrop-filter: blur(8px);
  transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); border-color: rgba(0,212,255,0.35); }

.kpi-value {
  font-family: 'Syne', sans-serif;
  font-size: 1.8rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--cyan), var(--violet));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1;
  margin-bottom: 0.3rem;
}

.kpi-label {
  font-size: 0.72rem;
  font-weight: 500;
  color: var(--text-md);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

/* ── Streamlit native components theming ── */
[data-testid="stTextArea"] textarea {
  background: rgba(7, 15, 30, 0.9) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-hi) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.95rem !important;
  transition: border-color 0.2s !important;
}
[data-testid="stTextArea"] textarea:focus {
  border-color: rgba(0,212,255,0.5) !important;
  box-shadow: 0 0 0 3px rgba(0,212,255,0.08) !important;
}

[data-testid="stSelectbox"] > div > div {
  background: rgba(7,15,30,0.9) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-hi) !important;
}

/* ── Buttons ── */
.stButton > button {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  border-radius: 10px !important;
  border: none !important;
  transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--cyan), var(--violet)) !important;
  color: #fff !important;
  box-shadow: 0 4px 20px rgba(0,212,255,0.25) !important;
}

.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 28px rgba(0,212,255,0.35) !important;
}

.stButton > button[kind="secondary"] {
  background: var(--glass) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-md) !important;
}

/* ── Translation output box ── */
.translated-box {
  background: linear-gradient(135deg, rgba(0,212,255,0.04), rgba(108,99,255,0.04));
  border: 1px solid rgba(0,212,255,0.25);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  font-size: 1.05rem;
  color: var(--text-hi);
  line-height: 1.7;
  min-height: 120px;
  white-space: pre-wrap;
  word-break: break-word;
  position: relative;
}

.translated-box::before {
  content: '✦ Translation';
  position: absolute;
  top: -0.6rem;
  left: 1rem;
  background: var(--bg1);
  padding: 0 0.5rem;
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--cyan);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
  margin: 0.75rem 0;
}
.conf-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.78rem;
  color: var(--text-md);
  margin-bottom: 4px;
}
.conf-bar {
  height: 6px;
  border-radius: 3px;
  background: var(--bg3);
  overflow: hidden;
}
.conf-fill {
  height: 100%;
  border-radius: 3px;
  background: linear-gradient(90deg, var(--cyan), var(--mint));
  transition: width 0.6s ease;
}

/* ── Section labels ── */
.section-label {
  font-family: 'Syne', sans-serif;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--cyan);
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Pill chips ── */
.chip {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.25rem 0.7rem;
  border-radius: 50px;
  font-size: 0.75rem;
  font-weight: 500;
  border: 1px solid;
}
.chip-cyan  { color: var(--cyan);   border-color: rgba(0,212,255,0.35);  background: rgba(0,212,255,0.06); }
.chip-violet{ color: var(--violet); border-color: rgba(108,99,255,0.35); background: rgba(108,99,255,0.06); }
.chip-mint  { color: var(--mint);   border-color: rgba(0,255,178,0.35);  background: rgba(0,255,178,0.06); }
.chip-rose  { color: var(--rose);   border-color: rgba(255,77,109,0.35); background: rgba(255,77,109,0.06); }

/* ── Pipeline diagram ── */
.pipeline {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  padding: 1rem 0;
}
.pipeline-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.4rem;
  flex: 1;
  min-width: 90px;
}
.pipeline-icon {
  width: 52px;
  height: 52px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.4rem;
  border: 1px solid var(--border);
  background: var(--glass);
}
.pipeline-arrow {
  font-size: 1.2rem;
  color: var(--text-lo);
  flex: 0;
  padding-bottom: 1.4rem;
}
.pipeline-name {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text-md);
  text-align: center;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* ── History table tweaks ── */
[data-testid="stDataFrame"] {
  border-radius: 10px !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
}

/* ── Audio player ── */
audio { width: 100%; border-radius: 8px; }

/* ── Divider ── */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
  margin: 1.5rem 0;
}

/* ── Metric row ── */
.metric-row {
  display: flex;
  justify-content: space-between;
  padding: 0.6rem 0;
  border-bottom: 1px solid var(--glass);
  font-size: 0.85rem;
}
.metric-key { color: var(--text-md); }
.metric-val { color: var(--text-hi); font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

/* ── NLP word cloud placeholder ── */
.word-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  padding: 0.75rem 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.3); }

/* ── Hide Streamlit footer ── */
footer { display: none !important; }

/* ── Spinner color ── */
[data-testid="stSpinner"] > div { border-top-color: var(--cyan) !important; }

/* ── Info / warning / success boxes ── */
[data-testid="stAlert"] {
  border-radius: 10px !important;
  border-left-color: var(--cyan) !important;
  background: rgba(0,212,255,0.05) !important;
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════════

if "history" not in st.session_state:
    st.session_state.history = []

if "last_translation" not in st.session_state:
    st.session_state.last_translation = None

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _conf_bar(confidence: float, label: str = "Confidence"):
    pct = int(confidence * 100)
    color_class = "rgba(0,255,178,1)" if pct >= 85 else ("rgba(0,212,255,1)" if pct >= 65 else "rgba(255,77,109,1)")
    st.markdown(f"""
    <div class="conf-bar-wrap">
      <div class="conf-label"><span>{label}</span><span style="color:{color_class};font-weight:600">{pct}%</span></div>
      <div class="conf-bar"><div class="conf-fill" style="width:{pct}%;background:linear-gradient(90deg,{color_class},var(--cyan));"></div></div>
    </div>
    """, unsafe_allow_html=True)


def _kpi(value, label: str):
    return f"""
    <div class="kpi-card">
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>
    """


def _chip(text: str, cls: str = "chip-cyan"):
    return f'<span class="chip {cls}">{text}</span>'


def _section(label: str, icon: str = ""):
    st.markdown(f'<div class="section-label">{icon} {label}</div>', unsafe_allow_html=True)


def history_to_df() -> pd.DataFrame:
    if not st.session_state.history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.history)


def _lang_name(code: str) -> str:
    return CODE_TO_NAME.get(code, code.upper())


# ═══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="hero-globe">🌍</div>
  <h1 class="hero-title">LinguaAI</h1>
  <p class="hero-sub">AI-Powered Intelligent Multilingual Translation Platform</p>
  <div class="hero-badges">
    <span class="badge active">✦ 100+ Languages</span>
    <span class="badge">🧠 NLP Analytics</span>
    <span class="badge">🔊 Text-to-Speech</span>
    <span class="badge">⚡ Real-time</span>
    <span class="badge">🎯 Auto-Detect</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡ Smart Translator",
    "🧠 AI Insights",
    "🔊 Text-to-Speech",
    "📋 History",
    "ℹ️ About AI",
])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — SMART TRANSLATOR
# ───────────────────────────────────────────────────────────────────────────────
with tab1:

    # Language selectors row
    col_src, col_arrow, col_tgt = st.columns([5, 1, 5])

    with col_src:
        _section("Source Language", "🌐")
        src_lang_name = st.selectbox(
            "Source Language",
            options=list(SOURCE_LANGUAGES.keys()),
            index=0,
            label_visibility="collapsed",
            key="src_lang_select",
        )
        src_code = SOURCE_LANGUAGES[src_lang_name]

    with col_arrow:
        st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;font-size:1.5rem;color:var(--cyan);padding-top:1.5rem'>⇄</div>", unsafe_allow_html=True)

    with col_tgt:
        _section("Target Language", "🎯")
        tgt_lang_list = list(TARGET_LANGUAGES.keys())
        default_tgt_idx = tgt_lang_list.index("Spanish") if "Spanish" in tgt_lang_list else 1
        tgt_lang_name = st.selectbox(
            "Target Language",
            options=tgt_lang_list,
            index=default_tgt_idx,
            label_visibility="collapsed",
            key="tgt_lang_select",
        )
        tgt_code = TARGET_LANGUAGES[tgt_lang_name]

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Input area
    col_in, col_out = st.columns(2)

    with col_in:
        _section("Input Text", "✏️")
        input_text = st.text_area(
            "Input",
            value=st.session_state.input_text,
            height=220,
            placeholder="Type or paste text to translate...\n\nSupports 100+ languages with auto-detection.",
            label_visibility="collapsed",
            key="main_input",
        )
        char_cnt = len(input_text)
        word_cnt = len(input_text.split()) if input_text.strip() else 0
        st.markdown(
            f'<div style="text-align:right;font-size:0.75rem;color:var(--text-lo);margin-top:4px">'
            f'{char_cnt} chars · {word_cnt} words</div>',
            unsafe_allow_html=True,
        )

    with col_out:
        _section("Translation", "✦")
        if st.session_state.last_translation and st.session_state.last_translation.get("translated_text"):
            tr = st.session_state.last_translation
            st.markdown(
                f'<div class="translated-box">{tr["translated_text"]}</div>',
                unsafe_allow_html=True,
            )
            _conf_bar(tr.get("confidence", 0.85))
            st.markdown(
                f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.5rem">'
                f'{_chip("⚙️ " + tr.get("engine","Google Translate"), "chip-cyan")}'
                f'{_chip("⏱ " + str(tr.get("time_taken","—")) + "s", "chip-violet")}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="translated-box" style="color:var(--text-lo);font-size:0.9rem;">'
                'Translation will appear here…</div>',
                unsafe_allow_html=True,
            )

    # Action buttons
    bcol1, bcol2, bcol3, bcol4 = st.columns([3, 2, 2, 2])

    with bcol1:
        translate_btn = st.button("⚡ Translate Now", type="primary", use_container_width=True)

    with bcol2:
        clear_btn = st.button("✕ Clear", type="secondary", use_container_width=True)

    with bcol3:
        detect_btn = st.button("🎯 Detect Lang", type="secondary", use_container_width=True)

    with bcol4:
        # Copy to clipboard via JS workaround
        if st.session_state.last_translation and st.session_state.last_translation.get("translated_text"):
            txt_b64 = base64.b64encode(
                st.session_state.last_translation["translated_text"].encode()
            ).decode()
            copy_js = f"""
            <script>
            function copyText() {{
              const text = atob("{txt_b64}");
              navigator.clipboard.writeText(text).then(() => {{
                const btn = document.getElementById('copy-btn');
                btn.innerText = '✓ Copied!';
                btn.style.color = 'var(--mint)';
                setTimeout(() => {{ btn.innerText = '⧉ Copy'; btn.style.color = ''; }}, 1500);
              }});
            }}
            </script>
            <button id="copy-btn"
              onclick="copyText()"
              style="width:100%;padding:.45rem .8rem;background:rgba(255,255,255,0.04);
                     border:1px solid rgba(0,212,255,0.2);border-radius:10px;
                     color:var(--text-md);cursor:pointer;font-family:'DM Sans',sans-serif;
                     font-size:.875rem;transition:all .2s">⧉ Copy</button>
            """
            st.markdown(copy_js, unsafe_allow_html=True)

    # ── Handle actions ─────────────────────────────────────────────────────────

    if clear_btn:
        st.session_state.input_text = ""
        st.session_state.last_translation = None
        st.rerun()

    if detect_btn and input_text.strip():
        with st.spinner("Detecting language..."):
            det = detect_language(input_text)
        if det.get("language_name"):
            st.success(f"🎯 Detected: **{det['language_name']}** ({det['language_code'].upper()}) — Confidence: {int(det['confidence']*100)}%")
        else:
            st.warning("Could not detect language.")

    if translate_btn:
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to translate.")
        elif src_code == tgt_code and src_code != "auto":
            st.info("ℹ️ Source and target languages are the same.")
        else:
            with st.spinner("Translating…"):
                result = translate_text(input_text, src_code, tgt_code)

            if result.get("error") and not result.get("translated_text"):
                st.error(f"❌ Translation failed: {result['error']}")
            else:
                st.session_state.last_translation = result

                # Add to history
                det_lang = detect_language(input_text)
                history_entry = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "source_lang": det_lang.get("language_name", src_lang_name),
                    "target_lang": tgt_lang_name,
                    "original": input_text[:300],
                    "translated": result.get("translated_text", "")[:300],
                    "confidence": f"{int(result.get('confidence', 0)*100)}%",
                    "engine": result.get("engine", ""),
                    "time_s": result.get("time_taken", 0),
                }
                st.session_state.history.insert(0, history_entry)
                # Keep last 100 entries
                st.session_state.history = st.session_state.history[:100]
                st.rerun()

    # ── Download translated text ───────────────────────────────────────────────
    if st.session_state.last_translation and st.session_state.last_translation.get("translated_text"):
        tr = st.session_state.last_translation
        dl_text = (
            f"LinguaAI – Translation Result\n"
            f"{'='*50}\n"
            f"Original  : {tr.get('original_text','')}\n"
            f"Translated: {tr.get('translated_text','')}\n"
            f"From      : {src_lang_name}\n"
            f"To        : {tgt_lang_name}\n"
            f"Engine    : {tr.get('engine','')}\n"
            f"Confidence: {int(tr.get('confidence',0)*100)}%\n"
            f"Time      : {tr.get('time_taken',0)}s\n"
            f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        st.download_button(
            "⬇ Download Translation",
            data=dl_text.encode("utf-8"),
            file_name=f"linguaai_translation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            type="secondary",
        )


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — AI INSIGHTS
# ───────────────────────────────────────────────────────────────────────────────
with tab2:

    input_for_analysis = st.session_state.get("main_input", "") or ""

    if not input_for_analysis.strip():
        st.info("💡 Enter text in the **Smart Translator** tab to see AI Insights.")
    else:
        # Run analysis
        with st.spinner("Running NLP analysis..."):
            analysis = analyze_text(input_for_analysis)
            det_result = detect_language(input_for_analysis)

        # ── Detection result ──────────────────────────────────────────────────
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        _section("Language Detection", "🎯")

        det_cols = st.columns([2, 1, 1])
        with det_cols[0]:
            lang_name = det_result.get("language_name", "Unknown")
            lang_code = det_result.get("language_code", "—")
            conf = det_result.get("confidence", 0)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1rem;padding:.5rem 0">
              <div style="font-size:2rem">🌐</div>
              <div>
                <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--cyan)">{lang_name}</div>
                <div style="font-size:.75rem;color:var(--text-md);font-family:'JetBrains Mono',monospace;margin-top:2px">ISO 639: {lang_code.upper()}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            _conf_bar(conf, "Detection Confidence")

        with det_cols[1]:
            alts = det_result.get("alternatives", [])
            if alts:
                st.markdown("**Alternatives**")
                for alt in alts[:3]:
                    st.markdown(
                        f'<div style="margin:.3rem 0">'
                        f'{_chip(alt["name"], "chip-violet")} '
                        f'<span style="font-size:.72rem;color:var(--text-lo)">{int(alt["confidence"]*100)}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with det_cols[2]:
            tr = st.session_state.last_translation
            if tr and tr.get("engine"):
                st.markdown("**Last Translation**")
                st.markdown(
                    f'<div style="margin:.3rem 0">{_chip("⚙️ " + tr["engine"], "chip-cyan")}</div>'
                    f'<div style="margin:.3rem 0">{_chip("⏱ " + str(tr.get("time_taken","—")) + "s", "chip-mint")}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── KPI cards ─────────────────────────────────────────────────────────
        _section("Text Statistics", "📊")

        kpi_html = '<div class="kpi-grid">'
        kpi_html += _kpi(analysis.get("char_count", 0), "Characters")
        kpi_html += _kpi(analysis.get("word_count", 0), "Words")
        kpi_html += _kpi(analysis.get("sentence_count", 0), "Sentences")
        kpi_html += _kpi(analysis.get("unique_words", 0), "Unique Words")
        kpi_html += _kpi(f"{analysis.get('type_token_ratio', 0):.2f}", "Lexical Diversity")
        kpi_html += _kpi(f"{analysis.get('avg_word_length', 0):.1f}", "Avg Word Length")
        kpi_html += _kpi(f"{analysis.get('avg_sentence_length', 0):.1f}", "Avg Sent. Length")
        kpi_html += _kpi(f"G{analysis.get('estimated_reading_level', 0):.0f}", "Reading Level")
        kpi_html += '</div>'
        st.markdown(kpi_html, unsafe_allow_html=True)

        # ── Top words ─────────────────────────────────────────────────────────
        top_words = analysis.get("top_words", [])
        if top_words:
            _section("Top Keywords", "🔑")
            whtml = '<div class="word-list">'
            colors = ["chip-cyan", "chip-violet", "chip-mint", "chip-rose", "chip-cyan"]
            for i, w in enumerate(top_words):
                whtml += f'<span class="chip {colors[i%5]}">{w["word"]} <span style="opacity:.6">×{w["count"]}</span></span>'
            whtml += '</div>'
            st.markdown(whtml, unsafe_allow_html=True)

        # ── Detailed metrics table ────────────────────────────────────────────
        with st.expander("📋 Full NLP Report", expanded=False):
            metrics = [
                ("Total Characters", analysis.get("char_count", 0)),
                ("Characters (no spaces)", analysis.get("char_no_spaces", 0)),
                ("Word Count", analysis.get("word_count", 0)),
                ("Sentence Count", analysis.get("sentence_count", 0)),
                ("Token Count", analysis.get("token_count", 0)),
                ("Unique Words", analysis.get("unique_words", 0)),
                ("Type-Token Ratio", f"{analysis.get('type_token_ratio', 0):.3f}"),
                ("Avg Word Length", f"{analysis.get('avg_word_length', 0):.2f} chars"),
                ("Avg Sentence Length", f"{analysis.get('avg_sentence_length', 0):.1f} words"),
                ("Punctuation Density", f"{analysis.get('punctuation_density', 0):.2f}%"),
                ("Est. Reading Level", f"Grade {analysis.get('estimated_reading_level', 0):.0f}"),
            ]
            rows_html = "".join(
                f'<div class="metric-row"><span class="metric-key">{k}</span><span class="metric-val">{v}</span></div>'
                for k, v in metrics
            )
            st.markdown(f'<div class="glass-card-sm">{rows_html}</div>', unsafe_allow_html=True)

        # ── NLP Pipeline flow ─────────────────────────────────────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        _section("NLP Processing Pipeline", "⚙️")
        pipeline_html = """
        <div class="pipeline">
          <div class="pipeline-step">
            <div class="pipeline-icon">📥</div>
            <div class="pipeline-name">Input</div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step">
            <div class="pipeline-icon">🔍</div>
            <div class="pipeline-name">Detect Lang</div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step">
            <div class="pipeline-icon">✂️</div>
            <div class="pipeline-name">Tokenize</div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step">
            <div class="pipeline-icon">⚙️</div>
            <div class="pipeline-name">Preprocess</div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step">
            <div class="pipeline-icon">🧠</div>
            <div class="pipeline-name">NMT Engine</div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step">
            <div class="pipeline-icon">📊</div>
            <div class="pipeline-name">Score</div>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step">
            <div class="pipeline-icon">📤</div>
            <div class="pipeline-name">Output</div>
          </div>
        </div>
        """
        st.markdown(pipeline_html, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — TEXT-TO-SPEECH
# ───────────────────────────────────────────────────────────────────────────────
with tab3:

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    _section("Speech Synthesis", "🔊")

    tts_col1, tts_col2 = st.columns([3, 2])

    with tts_col1:
        tts_text = st.text_area(
            "Text to speak",
            height=180,
            placeholder="Enter text to convert to speech…",
            label_visibility="visible",
            key="tts_input",
        )

        # Prefill from last translation
        if st.session_state.last_translation and st.session_state.last_translation.get("translated_text"):
            if st.button("⬆ Use Last Translation", type="secondary"):
                st.session_state["tts_input"] = st.session_state.last_translation["translated_text"]
                st.rerun()

    with tts_col2:
        tts_langs = get_tts_languages()
        tts_lang_options = [l["name"] for l in tts_langs]
        tts_default = tts_lang_options.index("English") if "English" in tts_lang_options else 0
        selected_tts_lang_name = st.selectbox("Language", tts_lang_options, index=tts_default, key="tts_lang")
        selected_tts_code = next(
            (l["code"] for l in tts_langs if l["name"] == selected_tts_lang_name), "en"
        )
        slow_mode = st.checkbox("🐢 Slow mode", value=False, key="tts_slow")

        st.markdown(f"""
        <div class="glass-card-sm" style="margin-top:1rem">
          <div class="metric-row"><span class="metric-key">Language</span><span class="metric-val">{selected_tts_lang_name}</span></div>
          <div class="metric-row"><span class="metric-key">Code</span><span class="metric-val">{selected_tts_code}</span></div>
          <div class="metric-row"><span class="metric-key">Engine</span><span class="metric-val">gTTS</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    gen_btn = st.button("🔊 Generate Speech", type="primary", use_container_width=False)

    if gen_btn:
        if not tts_text.strip():
            st.warning("Please enter some text for speech synthesis.")
        else:
            with st.spinner("Synthesizing speech…"):
                tts_result = text_to_speech(tts_text, selected_tts_code, slow=slow_mode)

            if tts_result.get("audio_bytes"):
                audio_bytes = tts_result["audio_bytes"]
                st.success(
                    f"✅ Speech generated — Language: {selected_tts_lang_name} "
                    f"· Est. duration: ~{tts_result.get('duration_estimate', 0):.0f}s"
                )
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button(
                    "⬇ Download MP3",
                    data=audio_bytes,
                    file_name=f"linguaai_speech_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                    mime="audio/mp3",
                    type="secondary",
                )
                if tts_result.get("error"):
                    st.info(f"ℹ️ {tts_result['error']}")
            else:
                st.error(f"❌ TTS failed: {tts_result.get('error', 'Unknown error')}")


# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 — TRANSLATION HISTORY
# ───────────────────────────────────────────────────────────────────────────────
with tab4:

    if not st.session_state.history:
        st.info("📋 No translations yet. Use the **Smart Translator** to get started.")
    else:
        hist_col1, hist_col2 = st.columns([4, 1])
        with hist_col1:
            _section(f"Translation History ({len(st.session_state.history)} entries)", "📋")
        with hist_col2:
            if st.button("🗑 Clear All", type="secondary"):
                st.session_state.history = []
                st.rerun()

        # Search
        search_query = st.text_input("🔍 Search history…", placeholder="Search by text, language…", label_visibility="collapsed")

        df = history_to_df()
        if search_query.strip():
            mask = df.apply(lambda row: search_query.lower() in str(row).lower(), axis=1)
            df = df[mask]

        if not df.empty:
            # Show summary KPIs
            kpis_h = '<div class="kpi-grid">'
            kpis_h += _kpi(len(st.session_state.history), "Total Translations")
            langs_used = set(r["target_lang"] for r in st.session_state.history)
            kpis_h += _kpi(len(langs_used), "Languages Used")
            avg_time = sum(r.get("time_s", 0) for r in st.session_state.history) / max(1, len(st.session_state.history))
            kpis_h += _kpi(f"{avg_time:.2f}s", "Avg Response Time")
            kpis_h += '</div>'
            st.markdown(kpis_h, unsafe_allow_html=True)

            # Display table
            display_cols = ["timestamp", "source_lang", "target_lang", "original", "translated", "confidence", "engine"]
            display_df = df[display_cols].copy() if all(c in df.columns for c in display_cols) else df
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Download CSV
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download History CSV",
                data=csv_data,
                file_name=f"linguaai_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="secondary",
            )
        else:
            st.info("No matching results found.")


# ───────────────────────────────────────────────────────────────────────────────
# TAB 5 — ABOUT AI
# ───────────────────────────────────────────────────────────────────────────────
with tab5:

    st.markdown("""
    <div class="glass-card">
      <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;
                  background:linear-gradient(135deg,var(--cyan),var(--violet));
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;margin-bottom:1rem">
        The Science Behind LinguaAI
      </div>
      <p style="color:var(--text-md);line-height:1.8;font-size:.95rem">
        LinguaAI combines <strong style="color:var(--cyan)">Neural Machine Translation (NMT)</strong>,
        <strong style="color:var(--violet)">Natural Language Processing (NLP)</strong>, and
        <strong style="color:var(--mint)">AI-powered language detection</strong> to deliver
        accurate, real-time multilingual translations across 100+ languages.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Four concept cards ────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="glass-card">
          <div style="font-size:2rem;margin-bottom:.75rem">🧠</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:var(--cyan);margin-bottom:.5rem">
            Neural Machine Translation
          </div>
          <p style="color:var(--text-md);font-size:.88rem;line-height:1.75">
            NMT uses deep neural networks — specifically <strong>Transformer architectures</strong>
            with self-attention mechanisms — to learn contextual mappings between languages.
            Unlike rule-based systems, NMT learns patterns from millions of bilingual sentence pairs,
            capturing idiomatic expressions and contextual nuances.
          </p>
          <div style="margin-top:1rem">
            <span class="chip chip-cyan">Transformer</span>&nbsp;
            <span class="chip chip-violet">Attention</span>&nbsp;
            <span class="chip chip-mint">Seq2Seq</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
          <div style="font-size:2rem;margin-bottom:.75rem">🔍</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:var(--violet);margin-bottom:.5rem">
            Language Detection
          </div>
          <p style="color:var(--text-md);font-size:.88rem;line-height:1.75">
            Language identification uses <strong>n-gram frequency profiles</strong> and
            <strong>character distribution models</strong> to classify text. The
            <code style="color:var(--mint)">langdetect</code> library (ported from Google's
            language-detection algorithm) achieves &gt;99% accuracy on texts longer than 50 characters
            across 55+ languages.
          </p>
          <div style="margin-top:1rem">
            <span class="chip chip-violet">N-gram Models</span>&nbsp;
            <span class="chip chip-cyan">Naive Bayes</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
          <div style="font-size:2rem;margin-bottom:.75rem">✂️</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:var(--mint);margin-bottom:.5rem">
            NLP Processing Pipeline
          </div>
          <p style="color:var(--text-md);font-size:.88rem;line-height:1.75">
            Each translation request flows through a multi-stage NLP pipeline:
            <strong>tokenization</strong> splits text into semantic units,
            <strong>normalization</strong> standardizes encoding and punctuation,
            <strong>sentence segmentation</strong> handles boundary detection, and
            <strong>post-processing</strong> restores formatting.
          </p>
          <div style="margin-top:1rem">
            <span class="chip chip-mint">NLTK</span>&nbsp;
            <span class="chip chip-cyan">Tokenizer</span>&nbsp;
            <span class="chip chip-violet">Segmenter</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
          <div style="font-size:2rem;margin-bottom:.75rem">🔊</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:var(--rose);margin-bottom:.5rem">
            Text-to-Speech Synthesis
          </div>
          <p style="color:var(--text-md);font-size:.88rem;line-height:1.75">
            TTS converts translated text to natural-sounding audio using Google's
            <strong>neural TTS engine</strong> via gTTS. The system uses
            <strong>prosody modeling</strong> to apply natural intonation patterns,
            handling pitch, timing, and emphasis for 40+ languages.
          </p>
          <div style="margin-top:1rem">
            <span class="chip chip-rose">gTTS</span>&nbsp;
            <span class="chip chip-cyan">Prosody</span>&nbsp;
            <span class="chip chip-mint">WaveNet</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Architecture table ────────────────────────────────────────────────────
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    _section("Technology Stack", "🏗️")

    tech_data = {
        "Component": ["Translation Engine", "Language Detection", "NLP Analytics", "Text-to-Speech", "Web Framework", "Data Processing"],
        "Technology": ["deep-translator (Google)", "langdetect", "NLTK + custom", "gTTS", "Streamlit", "pandas + numpy"],
        "Purpose": [
            "Primary neural translation via Google NMT API",
            "Statistical language ID with n-gram profiling",
            "Tokenization, segmentation, metrics",
            "Speech synthesis & audio export",
            "Interactive web UI with reactive state",
            "History management & CSV export",
        ],
    }
    tech_df = pd.DataFrame(tech_data)
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

    # ── Model explanation ─────────────────────────────────────────────────────
    with st.expander("📖 How the Translation Model Works", expanded=False):
        st.markdown("""
        <div class="glass-card-sm">
          <p style="color:var(--text-md);font-size:.88rem;line-height:1.85">
            <strong style="color:var(--cyan)">Step 1 — Encoding:</strong>
            The source sentence is tokenized and converted to dense vector embeddings using
            subword tokenization (BPE / SentencePiece). Each token is encoded with
            positional information.<br><br>

            <strong style="color:var(--violet)">Step 2 — Self-Attention:</strong>
            The Transformer encoder applies multi-head self-attention, allowing each token
            to attend to all other tokens and build contextual representations that capture
            long-range dependencies.<br><br>

            <strong style="color:var(--mint)">Step 3 — Cross-Attention Decoding:</strong>
            The decoder generates target-language tokens autoregressively. At each step,
            cross-attention over the encoder's output guides the decoder to focus on
            relevant source segments.<br><br>

            <strong style="color:var(--amber)">Step 4 — Beam Search:</strong>
            Rather than greedy decoding, beam search maintains the top-k partial translations,
            selecting the globally most probable output sequence — improving fluency and accuracy.
          </p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1rem;border-top:1px solid var(--border);margin-top:3rem">
  <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
              background:linear-gradient(135deg,var(--cyan),var(--violet));
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;margin-bottom:.5rem">
    🌍 LinguaAI
  </div>
  <div style="font-size:.78rem;color:var(--text-lo)">
    Built with  using Python · Streamlit · deep-translator · gTTS · NLTK
    &nbsp;·&nbsp;
  </div>
</div>
""", unsafe_allow_html=True)
