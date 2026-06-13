"""
LinguaAI - Language Detection Module
Uses langdetect with confidence scoring and NLP analytics
"""

import re
import time
from typing import Optional

try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Language code → human name mapping (subset for detection output)
LANG_NAMES = {
    "af": "Afrikaans", "sq": "Albanian", "am": "Amharic", "ar": "Arabic",
    "hy": "Armenian", "az": "Azerbaijani", "eu": "Basque", "be": "Belarusian",
    "bn": "Bengali", "bs": "Bosnian", "bg": "Bulgarian", "ca": "Catalan",
    "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
    "hr": "Croatian", "cs": "Czech", "da": "Danish", "nl": "Dutch",
    "en": "English", "eo": "Esperanto", "et": "Estonian", "fi": "Finnish",
    "fr": "French", "gl": "Galician", "ka": "Georgian", "de": "German",
    "el": "Greek", "gu": "Gujarati", "ht": "Haitian Creole", "hi": "Hindi",
    "hu": "Hungarian", "is": "Icelandic", "id": "Indonesian", "ga": "Irish",
    "it": "Italian", "ja": "Japanese", "kn": "Kannada", "ko": "Korean",
    "lv": "Latvian", "lt": "Lithuanian", "mk": "Macedonian", "ms": "Malay",
    "ml": "Malayalam", "mt": "Maltese", "mr": "Marathi", "mn": "Mongolian",
    "ne": "Nepali", "no": "Norwegian", "fa": "Persian", "pl": "Polish",
    "pt": "Portuguese", "pa": "Punjabi", "ro": "Romanian", "ru": "Russian",
    "sk": "Slovak", "sl": "Slovenian", "so": "Somali", "es": "Spanish",
    "sw": "Swahili", "sv": "Swedish", "tl": "Tagalog", "ta": "Tamil",
    "te": "Telugu", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese", "cy": "Welsh", "yo": "Yoruba",
    "zh": "Chinese",
}


def detect_language(text: str) -> dict:
    """
    Detect language of input text.
    Returns dict with: language_code, language_name, confidence, alternatives
    """
    result = {
        "language_code": "unknown",
        "language_name": "Unknown",
        "confidence": 0.0,
        "alternatives": [],
        "error": None,
    }

    if not text or len(text.strip()) < 3:
        result["error"] = "Text too short for reliable detection"
        return result

    if not LANGDETECT_AVAILABLE:
        result["error"] = "langdetect not installed"
        return result

    try:
        lang_probs = detect_langs(text)
        if lang_probs:
            top = lang_probs[0]
            code = top.lang
            result["language_code"] = code
            result["language_name"] = LANG_NAMES.get(code, code.upper())
            result["confidence"] = round(top.prob, 3)
            result["alternatives"] = [
                {
                    "code": l.lang,
                    "name": LANG_NAMES.get(l.lang, l.lang.upper()),
                    "confidence": round(l.prob, 3),
                }
                for l in lang_probs[1:4]
            ]
    except Exception as e:
        # Fallback to simple detect
        try:
            code = detect(text)
            result["language_code"] = code
            result["language_name"] = LANG_NAMES.get(code, code.upper())
            result["confidence"] = 0.85
        except Exception as e2:
            result["error"] = str(e2)

    return result


def analyze_text(text: str) -> dict:
    """
    Full NLP analytics for input text.
    Returns word count, char count, sentences, tokens, etc.
    """
    if not text:
        return {}

    # Basic metrics
    char_count = len(text)
    char_no_spaces = len(text.replace(" ", ""))
    words = text.split()
    word_count = len(words)

    # Sentence segmentation
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
            tokens = word_tokenize(text)
        except Exception:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            tokens = words
    else:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        tokens = words

    sentence_count = max(1, len(sentences))

    # Lexical diversity (type-token ratio)
    unique_words = set(w.lower() for w in words if w.isalpha())
    alpha_words = [w for w in words if w.isalpha()]
    ttr = round(len(unique_words) / max(1, len(alpha_words)), 3)

    # Average word length
    avg_word_len = round(
        sum(len(w) for w in words if w.isalpha()) / max(1, len(alpha_words)), 2
    )

    # Avg sentence length
    avg_sent_len = round(word_count / sentence_count, 1)

    # Reading level estimate (Flesch-Kincaid simplified)
    syllables = sum(_count_syllables(w) for w in alpha_words)
    avg_syllables = syllables / max(1, len(alpha_words))
    fk_grade = round(0.39 * avg_sent_len + 11.8 * avg_syllables - 15.59, 1)
    fk_grade = max(1, min(18, fk_grade))

    # Punctuation density
    punct_count = sum(1 for c in text if c in '.,!?;:"-\'()')
    punct_density = round(punct_count / max(1, char_count) * 100, 2)

    return {
        "char_count": char_count,
        "char_no_spaces": char_no_spaces,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "token_count": len(tokens),
        "unique_words": len(unique_words),
        "type_token_ratio": ttr,
        "avg_word_length": avg_word_len,
        "avg_sentence_length": avg_sent_len,
        "estimated_reading_level": fk_grade,
        "punctuation_density": punct_density,
        "top_words": _top_words(words, n=5),
    }


def _count_syllables(word: str) -> int:
    """Approximate syllable count for a word."""
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _top_words(words: list, n: int = 5) -> list:
    """Return top N most frequent meaningful words."""
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "i", "you", "he",
        "she", "it", "we", "they", "this", "that", "these", "those",
    }
    freq = {}
    for w in words:
        w_clean = w.lower().strip(".,!?;:")
        if w_clean.isalpha() and w_clean not in stopwords and len(w_clean) > 2:
            freq[w_clean] = freq.get(w_clean, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [{"word": w, "count": c} for w, c in sorted_words[:n]]
