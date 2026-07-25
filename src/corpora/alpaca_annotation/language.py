"""Deterministic language detection for output splitting."""

from __future__ import annotations

import re

from .models import Language


_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_BASE64ISH_RE = re.compile(r"^[A-Za-z0-9+/=_-]{12,}$")


def detect_language(*texts: str) -> Language:
    """Classify text as en, zh, multilingual, or unknown.

    The heuristic is intentionally conservative because the value controls file
    placement, not model behavior.
    """
    combined = " ".join(t for t in texts if t).strip()
    if len(combined) < 8:
        return "unknown"

    cjk_count = len(_CJK_RE.findall(combined))
    latin_words = _LATIN_WORD_RE.findall(combined)
    latin_chars = sum(len(word) for word in latin_words)
    alnum_count = sum(ch.isalnum() for ch in combined)

    if cjk_count == 0 and len(latin_words) == 0:
        return "unknown"
    if cjk_count == 0 and _BASE64ISH_RE.fullmatch(combined) and " " not in combined:
        return "unknown"

    total_signal = cjk_count + latin_chars
    if total_signal == 0:
        return "unknown"

    cjk_ratio = cjk_count / total_signal
    latin_ratio = latin_chars / total_signal

    if cjk_ratio >= 0.2 and latin_ratio >= 0.2 and cjk_count >= 4 and len(latin_words) >= 2:
        return "multilingual"
    if cjk_count >= 4 and cjk_ratio >= 0.35:
        return "zh"
    if latin_chars >= 12 and latin_ratio >= 0.65 and alnum_count >= 12:
        return "en"
    return "unknown"
