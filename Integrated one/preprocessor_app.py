# preprocessor_app.py

import io
import pytesseract
import pdfplumber
import csv
import pandas as pd
from docx import Document
from PIL import Image
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from preprocessing.preprocess_input import preprocess_input_from_bytes
from preprocessing.logger import logger

# Tesseract path - update based on your OS
# macOS (Homebrew): /opt/homebrew/bin/tesseract or /usr/local/bin/tesseract
# Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Linux: usually auto-detected
import platform
if platform.system() == "Darwin":  # macOS
    import shutil
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="Trinetra Person-A Preprocessor")

# Enable CORS for frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (frontend)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# TEXT INPUT (LONG TEXT SUPPORTED)
# -------------------------------------------------------------
class TextInput(BaseModel):
    text: str


@app.post("/analyze_text")
async def analyze_text(payload: TextInput):
    """Accepts long plain text and returns only clean extracted text."""
    try:
        raw = payload.text.encode("utf-8")
        result = preprocess_input_from_bytes(raw, return_steps=False)
        return {"text": result["text"]}
    except Exception as e:
        logger.error(f"[analyze_text] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------------------------------------------
# IMAGE INPUT (ANY FORMAT) — OCR
# -------------------------------------------------------------
@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        result = preprocess_input_from_bytes(raw, filename=file.filename, return_steps=False)
        return {"text": result["text"]}
    except Exception as e:
        logger.error(f"[analyze_image] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------------------------------------------
# PDF INPUT — TEXT + OCR FALLBACK
# -------------------------------------------------------------
@app.post("/analyze_pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            return JSONResponse({"error": "File must be a PDF"}, status_code=400)

        raw = await file.read()
        extracted_text = ""

        # Try PDF text extraction
        try:
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                for page in pdf.pages:
                    extracted_text += (page.extract_text() or "") + "\n"
        except Exception as e:
            logger.error(f"[pdfplumber error] {e}")

        # OCR fallback
        if not extracted_text.strip():
            try:
                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    for page in pdf.pages:
                        img = page.to_image(resolution=300).original
                        ocr_text = pytesseract.image_to_string(img)
                        extracted_text += ocr_text + "\n"
            except Exception as e:
                logger.error(f"[PDF OCR error] {e}")

        result = preprocess_input_from_bytes(extracted_text.encode("utf-8"), return_steps=False)
        return {"text": result["text"]}

    except Exception as e:
        logger.error(f"[analyze_pdf] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------------------------------------------
# DOCUMENT INPUT: CSV, DOCX, TXT, JSON, LOG, EXCEL, ANY TEXT FILE
# -------------------------------------------------------------
@app.post("/analyze_document")
async def analyze_document(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()
        raw = await file.read()

        # ---------------- CSV ----------------
        if filename.endswith(".csv"):
            try:
                text_output = ""
                csv_file = io.StringIO(raw.decode("utf-8", errors="ignore"))
                reader = csv.reader(csv_file)
                for row in reader:
                    text_output += " | ".join(row) + "\n"
                result = preprocess_input_from_bytes(text_output.encode("utf-8"), return_steps=False)
                return {"text": result["text"]}
            except Exception as e:
                logger.error(f"[CSV error] {e}")

        # ---------------- EXCEL (.xlsx / .xls) ----------------
        if filename.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(io.BytesIO(raw))
                text_output = df.to_string(index=False)
                result = preprocess_input_from_bytes(text_output.encode("utf-8"), return_steps=False)
                return {"text": result["text"]}
            except Exception as e:
                logger.error(f"[EXCEL error] {e}")
                return JSONResponse({"error": f"Excel read error: {e}"}, status_code=500)

        # ---------------- DOCX ----------------
        if filename.endswith(".docx"):
            try:
                doc = Document(io.BytesIO(raw))
                text = "\n".join([p.text for p in doc.paragraphs])
                result = preprocess_input_from_bytes(text.encode("utf-8"), return_steps=False)
                return {"text": result["text"]}
            except Exception as e:
                logger.error(f"[DOCX error] {e}")

        # ---------------- ANY TEXT FILE ----------------
        if filename.endswith((".txt", ".md", ".log", ".json", ".py")):
            result = preprocess_input_from_bytes(raw, return_steps=False)
            return {"text": result["text"]}

        # ---------------- FALLBACK (AUTO-DETECT) ----------------
        result = preprocess_input_from_bytes(raw, return_steps=False)
        return {"text": result["text"]}

    except Exception as e:
        logger.error(f"[analyze_document] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------------------------------------------
# URL INPUT: HTML, PDF, IMAGE
# -------------------------------------------------------------
@app.post("/analyze_url")
async def analyze_url(url: str):
    """Extract text from ANY URL:
       - HTML pages
       - PDFs
       - Images (JPG/PNG/WEBP)
       - Fallback: raw bytes
    """

    try:
        # --- IMPORTANT: Fake Browser Headers to bypass 403 ---
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/118.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        response = requests.get(url, headers=headers, timeout=12)

        if response.status_code != 200:
            return {"error": f"Failed to fetch URL: {response.status_code}"}

        raw = response.content
        content_type = response.headers.get("content-type", "").lower()

        # -----------------------------------------------------
        # 1) HTML PAGE → extract visible text
        # -----------------------------------------------------
        if "text/html" in content_type:
            soup = BeautifulSoup(raw, "html.parser")
            text = soup.get_text(separator="\n")
            cleaned = preprocess_input_from_bytes(text.encode(), return_steps=False)
            return {"text": cleaned["text"]}

        # -----------------------------------------------------
        # 2) PDF FILE
        # -----------------------------------------------------
        if "application/pdf" in content_type:
            extracted = ""
            try:
                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    for page in pdf.pages:
                        extracted += (page.extract_text() or "") + "\n"
            except:
                pass

            cleaned = preprocess_input_from_bytes(
                extracted.encode(), return_steps=False
            )
            return {"text": cleaned["text"]}

        # -----------------------------------------------------
        # 3) IMAGE FROM URL → OCR
        # -----------------------------------------------------
        if any(fmt in content_type for fmt in ["image", "jpg", "jpeg", "png", "webp"]):
            img = Image.open(io.BytesIO(raw))
            text = pytesseract.image_to_string(img)
            cleaned = preprocess_input_from_bytes(text.encode(), return_steps=False)
            return {"text": cleaned["text"]}

        # -----------------------------------------------------
        # 4) FALLBACK: unknown content → try pipeline anyway
        # -----------------------------------------------------
        cleaned = preprocess_input_from_bytes(raw, return_steps=False)
        return {"text": cleaned["text"]}

    except Exception as e:
        logger.error(f"[analyze_url] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------------------------------------------
# QR/BARCODE INPUT
# -------------------------------------------------------------
@app.post("/analyze_qr")
async def analyze_qr(file: UploadFile = File(...)):
    try:
        from pyzxing import BarCodeReader
        
        raw = await file.read()

        # Save temporarily in memory
        img = Image.open(io.BytesIO(raw))
        temp_path = "/tmp/qr_temp.png"
        img.save(temp_path)

        reader = BarCodeReader()
        result = reader.decode(temp_path)

        if not result:
            return {"error": "No QR/Barcode detected"}

        decoded = result[0].get("parsed", "")

        cleaned = preprocess_input_from_bytes(decoded.encode(), return_steps=False)
        return {"text": cleaned["text"]}

    except ImportError:
        return JSONResponse({"error": "pyzxing not installed. Run: pip install pyzxing"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🔬 Trinetra Preprocessor API")
    print("   Endpoints: /analyze_text, /analyze_image, /analyze_pdf")
    print("              /analyze_document, /analyze_url, /analyze_qr")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8001)
