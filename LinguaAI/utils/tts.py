"""
LinguaAI - Text-to-Speech Module
Uses gTTS (Google Text-to-Speech) with multi-language support
"""

import io
import time
from typing import Optional

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# gTTS-supported language codes (subset that works reliably)
TTS_SUPPORTED_LANGS = {
    "af", "ar", "bg", "bn", "bs", "ca", "cs", "cy", "da", "de",
    "el", "en", "eo", "es", "et", "fi", "fr", "gu", "hi", "hr",
    "hu", "hy", "id", "is", "it", "ja", "jw", "km", "kn", "ko",
    "la", "lv", "mk", "ml", "mr", "my", "ne", "nl", "no", "pl",
    "pt", "ro", "ru", "si", "sk", "sq", "sr", "su", "sv", "sw",
    "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "zh-CN", "zh-TW",
}

# Map non-standard codes to gTTS-compatible codes
LANG_CODE_MAP = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "iw": "iw",
    "auto": "en",  # fallback
}


def text_to_speech(text: str, lang_code: str = "en", slow: bool = False) -> dict:
    """
    Convert text to speech using gTTS.
    Returns dict with: audio_bytes, lang_used, supported, error
    """
    result = {
        "audio_bytes": None,
        "lang_used": lang_code,
        "supported": False,
        "error": None,
        "duration_estimate": 0,
    }

    if not GTTS_AVAILABLE:
        result["error"] = "gTTS library not installed. Run: pip install gtts"
        return result

    if not text or not text.strip():
        result["error"] = "No text provided for speech synthesis"
        return result

    # Normalize lang code
    normalized = LANG_CODE_MAP.get(lang_code.lower(), lang_code)

    # Check support and find fallback
    supported_code = _find_supported_lang(normalized)
    result["lang_used"] = supported_code
    result["supported"] = supported_code != "en" or lang_code in ("en", "auto")

    # Truncate very long texts (gTTS has limits)
    text_to_speak = text[:3000] if len(text) > 3000 else text

    try:
        tts = gTTS(text=text_to_speak, lang=supported_code, slow=slow)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        result["audio_bytes"] = audio_data
        # Rough estimate: ~150 words/minute
        word_count = len(text_to_speak.split())
        result["duration_estimate"] = round(word_count / 150 * 60, 1)

    except Exception as e:
        result["error"] = f"TTS generation failed: {str(e)}"
        # Try with English as last resort
        if supported_code != "en":
            try:
                tts = gTTS(text=text_to_speak, lang="en", slow=slow)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                result["audio_bytes"] = buf.read()
                result["lang_used"] = "en"
                result["error"] = f"Used English TTS (original lang not supported)"
            except Exception as e2:
                result["error"] = f"TTS completely failed: {str(e2)}"

    return result


def _find_supported_lang(lang_code: str) -> str:
    """Find the best supported language code for gTTS."""
    if not lang_code:
        return "en"

    code_lower = lang_code.lower()

    # Direct match
    if code_lower in [l.lower() for l in TTS_SUPPORTED_LANGS]:
        for l in TTS_SUPPORTED_LANGS:
            if l.lower() == code_lower:
                return l

    # Prefix match (e.g., "zh" → "zh-CN")
    for supported in TTS_SUPPORTED_LANGS:
        if supported.lower().startswith(code_lower):
            return supported

    return "en"


def get_tts_languages() -> list:
    """Return list of languages supported by TTS."""
    from utils.translator import CODE_TO_NAME
    result = []
    for code in sorted(TTS_SUPPORTED_LANGS):
        name = CODE_TO_NAME.get(code, CODE_TO_NAME.get(code.lower(), code))
        result.append({"code": code, "name": name})
    return sorted(result, key=lambda x: x["name"])
