# preprocessing/text_cleaner.py

import unicodedata
import re
from .logger import logger

# Invisible → visible token mapping (Preserve meaning)
INVISIBLE_MAP = {
    "\u200b": "[ZWSP]",   # zero-width space
    "\u200c": "[ZWNJ]",   # zero-width non-joiner
    "\u200d": "[ZWJ]",    # zero-width joiner
    "\ufeff": "[BOM]",    # byte order mark
    "\u2060": "[WJ]",     # word joiner
}


def map_invisible(text: str) -> str:
    """Convert invisible characters into visible tokens without removing anything."""
    for ch, token in INVISIBLE_MAP.items():
        text = text.replace(ch, token)
    return text


def clean_text(text: str) -> str:
    """
    PERSON-A FINAL TEXT CLEANER:
    ------------------------------------
    ✔ Normalize unicode width (NFKC)
    ✔ Expose invisible characters as visible tokens
    ✔ Normalize whitespace safely
    ✖ Do NOT lowercase
    ✖ Do NOT remove characters
    ✖ Do NOT collapse repeated characters
    ✖ Do NOT sanitize content
    """

    logger.debug(f"[text_cleaner] INPUT: {text}")

    if not isinstance(text, str):
        text = str(text)

    # 1) Unicode normalization (keeps meaning intact)
    normalized = unicodedata.normalize("NFKC", text)

    # 2) Make invisible characters visible
    visible = map_invisible(normalized)

    # 3) Replace control characters (except newline/tab) with visible token
    # Do NOT delete them.
    visible = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        "[CTRL]",
        visible
    )

    # 4) Normalize whitespace (but do NOT erase newlines)
    # Replace tabs with a space
    visible = visible.replace("\t", " ")

    # Collapse long sequences of spaces but keep newlines intact
    visible = re.sub(r" {2,}", " ", visible)

    # Replace 3+ newlines with 2 (preserves layout, avoids abuse)
    visible = re.sub(r"\n{3,}", "\n\n", visible)

    result = visible.strip()

    logger.debug(f"[text_cleaner] OUTPUT: {result}")
    return result
