"""
LinguaAI - Translation Engine
Supports Google Translate via deep-translator with LibreTranslate fallback
"""

import time
import requests
from deep_translator import GoogleTranslator
from typing import Optional

# ─── Language Registry ───────────────────────────────────────────────────────

LANGUAGES = {
    "Auto Detect": "auto",
    "Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Arabic": "ar",
    "Armenian": "hy", "Azerbaijani": "az", "Basque": "eu", "Belarusian": "be",
    "Bengali": "bn", "Bosnian": "bs", "Bulgarian": "bg", "Catalan": "ca",
    "Cebuano": "ceb", "Chinese (Simplified)": "zh-CN", "Chinese (Traditional)": "zh-TW",
    "Corsican": "co", "Croatian": "hr", "Czech": "cs", "Danish": "da",
    "Dutch": "nl", "English": "en", "Esperanto": "eo", "Estonian": "et",
    "Finnish": "fi", "French": "fr", "Frisian": "fy", "Galician": "gl",
    "Georgian": "ka", "German": "de", "Greek": "el", "Gujarati": "gu",
    "Haitian Creole": "ht", "Hausa": "ha", "Hawaiian": "haw", "Hebrew": "iw",
    "Hindi": "hi", "Hmong": "hmn", "Hungarian": "hu", "Icelandic": "is",
    "Igbo": "ig", "Indonesian": "id", "Irish": "ga", "Italian": "it",
    "Japanese": "ja", "Javanese": "jw", "Kannada": "kn", "Kazakh": "kk",
    "Khmer": "km", "Kinyarwanda": "rw", "Korean": "ko", "Kurdish": "ku",
    "Kyrgyz": "ky", "Lao": "lo", "Latin": "la", "Latvian": "lv",
    "Lithuanian": "lt", "Luxembourgish": "lb", "Macedonian": "mk",
    "Malagasy": "mg", "Malay": "ms", "Malayalam": "ml", "Maltese": "mt",
    "Maori": "mi", "Marathi": "mr", "Mongolian": "mn", "Myanmar": "my",
    "Nepali": "ne", "Norwegian": "no", "Nyanja": "ny", "Odia": "or",
    "Pashto": "ps", "Persian": "fa", "Polish": "pl", "Portuguese": "pt",
    "Punjabi": "pa", "Romanian": "ro", "Russian": "ru", "Samoan": "sm",
    "Scots Gaelic": "gd", "Serbian": "sr", "Sesotho": "st", "Shona": "sn",
    "Sindhi": "sd", "Sinhala": "si", "Slovak": "sk", "Slovenian": "sl",
    "Somali": "so", "Spanish": "es", "Sundanese": "su", "Swahili": "sw",
    "Swedish": "sv", "Tagalog": "tl", "Tajik": "tg", "Tamil": "ta",
    "Tatar": "tt", "Telugu": "te", "Thai": "th", "Turkish": "tr",
    "Turkmen": "tk", "Ukrainian": "uk", "Urdu": "ur", "Uyghur": "ug",
    "Uzbek": "uz", "Vietnamese": "vi", "Welsh": "cy", "Xhosa": "xh",
    "Yiddish": "yi", "Yoruba": "yo", "Zulu": "zu",
}

SOURCE_LANGUAGES = {"Auto Detect": "auto", **{k: v for k, v in LANGUAGES.items() if k != "Auto Detect"}}
TARGET_LANGUAGES = {k: v for k, v in LANGUAGES.items() if k != "Auto Detect"}

CODE_TO_NAME = {v: k for k, v in LANGUAGES.items()}


def translate_text(text: str, source_lang: str, target_lang: str) -> dict:
    """
    Translate text using Google Translate (deep-translator) with LibreTranslate fallback.
    Returns dict with: translated_text, source_lang, confidence, engine, time_taken
    """
    if not text or not text.strip():
        return {"error": "Empty input text"}

    start_time = time.time()
    result = {
        "original_text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "translated_text": "",
        "detected_lang": None,
        "confidence": 0.0,
        "engine": "",
        "time_taken": 0.0,
        "error": None,
    }

    # ── Primary: deep-translator Google ──────────────────────────────────────
    try:
        src = "auto" if source_lang == "auto" else source_lang
        translator = GoogleTranslator(source=src, target=target_lang)
        translated = translator.translate(text)

        if translated:
            result["translated_text"] = translated
            result["engine"] = "Google Translate"
            result["confidence"] = _estimate_confidence(text, translated, source_lang, target_lang)
            result["time_taken"] = round(time.time() - start_time, 3)
            return result

    except Exception as e:
        result["error"] = f"Primary engine failed: {str(e)}"

    # ── Fallback: LibreTranslate ──────────────────────────────────────────────
    try:
        libre_result = _libre_translate(text, source_lang, target_lang)
        if libre_result:
            result["translated_text"] = libre_result
            result["engine"] = "LibreTranslate"
            result["confidence"] = _estimate_confidence(text, libre_result, source_lang, target_lang)
            result["error"] = None
            result["time_taken"] = round(time.time() - start_time, 3)
            return result
    except Exception as e:
        result["error"] = f"Both engines failed. Last error: {str(e)}"

    result["time_taken"] = round(time.time() - start_time, 3)
    return result


def _libre_translate(text: str, source: str, target: str) -> Optional[str]:
    """LibreTranslate public API fallback."""
    endpoints = [
        "https://libretranslate.de/translate",
        "https://translate.argosopentech.com/translate",
    ]
    src = "auto" if source == "auto" else source
    for url in endpoints:
        try:
            response = requests.post(
                url,
                json={"q": text, "source": src, "target": target, "format": "text"},
                timeout=8,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("translatedText", "")
        except Exception:
            continue
    return None


def _estimate_confidence(original: str, translated: str, source: str, target: str) -> float:
    """
    Heuristic confidence estimation based on translation quality signals.
    Returns value between 0.0 and 1.0.
    """
    if not translated or not original:
        return 0.0

    score = 0.85  # base confidence for successful translation

    # Length ratio check (translated shouldn't be wildly different)
    orig_len = len(original.split())
    trans_len = len(translated.split())
    if orig_len > 0:
        ratio = trans_len / orig_len
        if 0.3 <= ratio <= 4.0:
            score += 0.10
        else:
            score -= 0.15

    # Same language penalty (if input equals output, likely error)
    if original.strip().lower() == translated.strip().lower():
        score -= 0.40

    # Short texts have lower confidence
    if len(original.split()) < 3:
        score -= 0.10

    return max(0.0, min(1.0, round(score, 2)))
