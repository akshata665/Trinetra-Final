# preprocessing/handler.py

import io
import base64
import binascii
import json
import csv
import unicodedata
import re
from typing import Optional, Tuple, Dict, Any

import pdfplumber
from PIL import Image
import pytesseract
import pandas as pd
from urllib.parse import unquote


# -------------------------------------------------------------
# 1. Invisible → Visible mapping (DO NOT REMOVE CHARACTERS)
# -------------------------------------------------------------
INVISIBLE_MAP = {
    "\u200b": "[ZWSP]",   # zero‑width space
    "\u200c": "[ZWNJ]",   # zero‑width non‑joiner
    "\u200d": "[ZWJ]",    # zero‑width joiner
    "\ufeff": "[BOM]",    # byte order mark
    "\u2060": "[WJ]",     # word joiner
}


def map_invisible_to_tokens(text: str) -> str:
    """Replace invisible characters with visible tokens without deleting text."""
    for ch, token in INVISIBLE_MAP.items():
        text = text.replace(ch, token)
    return text


# -------------------------------------------------------------
# 2. Normalize unicode + spacing WITHOUT modifying meaning
# -------------------------------------------------------------
def normalize_text(text: str) -> str:
    # Normalize unicode width (full‑width → half‑width, homoglyphs → standard)
    text = unicodedata.normalize("NFKC", text)

    # Expose invisible chars
    text = map_invisible_to_tokens(text)

    # Replace control chars (except \n, \t) with a space (not delete)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+", " ", text)

    # Normalize overlapping spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce too many blank lines but keep formatting
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# -------------------------------------------------------------
# 3. Try decoding raw bytes (UTF‑8 / base64 / hex / URL)
# -------------------------------------------------------------
def try_decode_encodings(raw_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    # 1. Try direct UTF‑8
    try:
        s = raw_bytes.decode("utf-8")
        if any(c.isalpha() for c in s):
            return s, "utf8"
    except:
        pass

    # 2. Try Base64
    try:
        decoded = base64.b64decode(raw_bytes, validate=True)
        try:
            decoded_text = decoded.decode("utf-8")
            return decoded_text, "base64"
        except:
            pass
    except:
        pass

    # 3. Try hex
    try:
        unhex = binascii.unhexlify(raw_bytes.strip())
        try:
            decoded_hex = unhex.decode("utf-8")
            return decoded_hex, "hex"
        except:
            pass
    except:
        pass

    # 4. Try URL decoding
    try:
        s = raw_bytes.decode("utf-8", errors="ignore")
        ud = unquote(s)
        if ud != s:
            return ud, "url"
    except:
        pass

    return None, None


# -------------------------------------------------------------
# 4. Extract text from PDF
# -------------------------------------------------------------
def extract_text_from_pdf_bytes(raw_bytes: bytes) -> str:
    text_parts = []

    # Try direct PDF text extraction
    try:
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                text_parts.append(txt)
    except:
        pass

    joined = "\n\n".join(t for t in text_parts if t)

    # If no text found → OCR fallback
    if not joined.strip():
        try:
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                for page in pdf.pages:
                    im = page.to_image(resolution=200).original
                    ocr = pytesseract.image_to_string(im)
                    text_parts.append(ocr)
        except:
            pass

        joined = "\n\n".join(t for t in text_parts if t)

    return joined


# -------------------------------------------------------------
# 5. Extract text from images
# -------------------------------------------------------------
def extract_text_from_image_bytes(raw_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return pytesseract.image_to_string(img)
    except:
        return ""


# -------------------------------------------------------------
# 6. JSON flatten
# -------------------------------------------------------------
def flatten_json_bytes(raw_bytes: bytes) -> str:
    try:
        obj = json.loads(raw_bytes.decode("utf-8", errors="ignore"))

        def flatten(o):
            if isinstance(o, dict):
                return " ; ".join(f"{k}: {flatten(v)}" for k, v in o.items())
            if isinstance(o, list):
                return " ; ".join(flatten(x) for x in o)
            return str(o)

        return flatten(obj)

    except:
        return raw_bytes.decode("utf-8", errors="ignore")


# -------------------------------------------------------------
# 7. CSV flatten
# -------------------------------------------------------------
def flatten_csv_bytes(raw_bytes: bytes) -> str:
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
        rows = []
        for _, row in df.iterrows():
            rows.append(" | ".join(f"{col}={row[col]}" for col in df.columns))
        return "\n".join(rows)
    except:
        return raw_bytes.decode("utf-8", errors="ignore")


# -------------------------------------------------------------
# 8. MASTER FUNCTION — Detect input type + extract text
# -------------------------------------------------------------
def extract_text_from_input(raw: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
    result = {
        "original_bytes_b64": base64.b64encode(raw).decode("ascii"),
        "detected_type": None,
        "decoded_via": None,
        "extracted_text": ""
    }

    lower = (filename or "").lower()

    # -----------------------------
    # Detect by file extension
    # -----------------------------
    if lower.endswith(".pdf"):
        result["detected_type"] = "pdf"
        result["extracted_text"] = extract_text_from_pdf_bytes(raw)
        return result

    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
        result["detected_type"] = "image"
        result["extracted_text"] = extract_text_from_image_bytes(raw)
        return result

    if lower.endswith(".json"):
        result["detected_type"] = "json"
        result["extracted_text"] = flatten_json_bytes(raw)
        return result

    if lower.endswith((".csv", ".tsv")):
        result["detected_type"] = "csv"
        result["extracted_text"] = flatten_csv_bytes(raw)
        return result

    # -----------------------------
    # Try encoded text (base64, hex, URL)
    # -----------------------------
    decoded, method = try_decode_encodings(raw)
    if decoded is not None:
        result["decoded_via"] = method
        result["detected_type"] = method
        result["extracted_text"] = decoded
        return result

    # -----------------------------
    # Detect by magic bytes
    # -----------------------------
    if raw.startswith(b"%PDF"):
        result["detected_type"] = "pdf"
        result["extracted_text"] = extract_text_from_pdf_bytes(raw)
        return result

    if raw[:4] in (b"\x89PNG", b"\xFF\xD8\xFF\xE0", b"GIF8"):
        result["detected_type"] = "image"
        result["extracted_text"] = extract_text_from_image_bytes(raw)
        return result

    # -----------------------------
    # Fallback: treat as UTF‑8 text
    # -----------------------------
    try:
        txt = raw.decode("utf-8", errors="ignore")
        result["detected_type"] = "text"
        result["extracted_text"] = txt
        return result
    except:
        result["detected_type"] = "binary"
        result["extracted_text"] = ""
        return result
