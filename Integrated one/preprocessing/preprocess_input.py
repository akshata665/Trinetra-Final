# preprocessing/preprocess_input.py
import io
import json
import base64
import re
import unicodedata
from typing import Optional
from PIL import Image, UnidentifiedImageError
import pytesseract
from .logger import logger


# -------------------------------------------------------------
# Helper: Detect if bytes are base64
# -------------------------------------------------------------
def _is_base64(s: bytes) -> bool:
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


# -------------------------------------------------------------
# Image OCR using Pillow ONLY (Python 3.14 safe)
# -------------------------------------------------------------
def try_image_ocr(raw_bytes):
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img_type = img.format  # PNG / JPEG / WEBP...
        text = pytesseract.image_to_string(img)
        return {
            "detected_type": f"image/{img_type.lower()}",
            "decoded_via": "pytesseract",
            "extracted_text": text,
        }
    except UnidentifiedImageError:
        return None
    except Exception:
        return None


# -------------------------------------------------------------
# Main extractor
# -------------------------------------------------------------
def extract_text_from_input(raw_bytes: bytes, filename: Optional[str] = None):
    """
    Robust extraction pipeline supporting:
      • UTF‑8 text
      • JSON {"text": "..."}
      • Base64 → auto decode
      • Image → OCR
      • Binary fallback (latin-1 decode)
    """

    info = {
        "detected_type": None,
        "decoded_via": None,
        "original_bytes_b64": base64.b64encode(raw_bytes).decode("utf-8"),
        "extracted_text": "",
    }

    # ---------------------------------------------------------
    # 1) Try UTF‑8 text decoding
    # ---------------------------------------------------------
    try:
        txt = raw_bytes.decode("utf-8")
        stripped = txt.strip()

        # JSON input
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(txt)
                info["detected_type"] = "json"
                info["decoded_via"] = "utf-8-json"
                if isinstance(parsed, dict) and "text" in parsed:
                    info["extracted_text"] = str(parsed["text"])
                else:
                    info["extracted_text"] = txt
                return info
            except Exception:
                # not JSON → treat as plain text
                info["detected_type"] = "text"
                info["decoded_via"] = "utf-8"
                info["extracted_text"] = txt
                return info

        # Plain text
        info["detected_type"] = "text"
        info["decoded_via"] = "utf-8"
        info["extracted_text"] = txt
        return info

    except UnicodeDecodeError:
        pass

    # ---------------------------------------------------------
    # 2) Base64 decode detection
    # ---------------------------------------------------------
    if _is_base64(raw_bytes):
        try:
            decoded = base64.b64decode(raw_bytes)
            inner = extract_text_from_input(decoded, filename)
            inner["detected_type"] = "base64->" + inner["detected_type"]
            inner["decoded_via"] = "base64->" + inner["decoded_via"]
            inner["original_bytes_b64"] = base64.b64encode(raw_bytes).decode()
            return inner
        except Exception:
            pass

    # ---------------------------------------------------------
    # 3) Try PIL Image → OCR
    # ---------------------------------------------------------
    img_res = try_image_ocr(raw_bytes)
    if img_res:
        img_res["original_bytes_b64"] = base64.b64encode(raw_bytes).decode()
        return img_res

    # ---------------------------------------------------------
    # 4) Fallback decode with latin‑1
    # ---------------------------------------------------------
    try:
        txt = raw_bytes.decode("latin-1")
        info["detected_type"] = "binary_as_latin1"
        info["decoded_via"] = "latin-1"
        info["extracted_text"] = txt
        return info
    except Exception:
        pass

    # ---------------------------------------------------------
    # 5) Final fallback: unknown binary
    # ---------------------------------------------------------
    info["detected_type"] = "unknown"
    info["decoded_via"] = "none"
    info["extracted_text"] = ""
    return info


# -------------------------------------------------------------
# Invisible char mapping
# -------------------------------------------------------------
def map_invisible_to_tokens(text: str) -> str:
    mapping = {
        "\u200b": "<ZWSP>",
        "\ufeff": "<BOM>",
        "\u200e": "<LRM>",
        "\u200f": "<RLM>",
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

# -------------------------------------------------------------
# Heuristic spacing recovery (CRITICAL)
# -------------------------------------------------------------
def recover_missing_spaces(text: str) -> str:
    # Add space between lowercase-uppercase
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    # Add space between letter-letter when a big word forms
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)

    # Add space between digit-letter and letter-digit
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)

    return text


# -------------------------------------------------------------
# Unicode + layout normalization (FINAL)
# -------------------------------------------------------------
def normalize_text(text: str) -> str:
    # 1) Unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # 2) Recover missing spaces (MOST IMPORTANT)
    text = recover_missing_spaces(text)

    # 3) Fix punctuation spacing
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)

    # 4) Collapse spaces but keep paragraphs
    lines = [" ".join(line.split()) for line in text.splitlines()]

    return "\n".join(lines).strip()


# -------------------------------------------------------------
# Final preprocess function
# -------------------------------------------------------------
def preprocess_input_from_bytes(raw_bytes: bytes, filename: str = None, return_steps: bool = True):
    logger.info("=== PREPROCESS_INPUT START ===")

    steps = {}
    extracted = extract_text_from_input(raw_bytes, filename)

    steps["extraction"] = {
        "detected_type": extracted["detected_type"],
        "decoded_via": extracted["decoded_via"],
        "original_bytes_b64": extracted["original_bytes_b64"],
    }

    raw_text = extracted["extracted_text"]
    steps["raw_extracted_text"] = raw_text

    mapped = map_invisible_to_tokens(raw_text)
    steps["invisible_mapped"] = mapped

    normalized = normalize_text(mapped)
    steps["normalized"] = normalized

    result_full = {
        "original_bytes_b64": extracted["original_bytes_b64"],
        "detected_type": extracted["detected_type"],
        "decoded_via": extracted["decoded_via"],
        "extracted_text": raw_text,
        "invisible_mapped": mapped,
        "final_normalized": normalized,
    }

    result_clean = {
        "type": extracted["detected_type"],
        "text": normalized,
    }

    return result_full if return_steps else result_clean
