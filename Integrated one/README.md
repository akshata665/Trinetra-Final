# TRINETRA - AI Security Intelligence Platform

A prevention-first, agentic defense framework designed to secure Large Language Models (LLMs) against sophisticated threats. Trinetra provides a **Layer 0** security boundary that proactively detects, isolates, and mitigates malicious inputs before they reach your core LLMs.

<p align="center">
  <img src="assets/trinetra-logo.png" alt="Trinetra Banner" width="100%">
</p>

## 🚀 Key Features

### 🛡️ Layer 0: Prompt Injection Guard
- **Heuristic Detection**: regex-based pattern matching (Groups A-E).
- **Semantic Analysis**: DeBERTa-v3 model (`protectai/deberta-v3-base-prompt-injection`) integration.
- **Weighted Risk Scoring**: Accumulate risk from multiple signals (e.g., `roleplay(+5) + override(+10) = BLOCK`).
- **2-Phase Architecture**:
    - **Phase 1 (Heuristic Limit)**: Instant regex check.
    - **Phase 2 (Semantic Check)**: ML-based intent analysis.

### ⚡ Proactive UI
- **Real-time Scanning**: Keystroke-level highlighting of dangerous terms (Yellow/Red).
- **Interactive Feedback**: Dynamic button states (`RUN DIAGNOSTICS` vs `🚫 BLOCKED`).
- **Minimalist Design**: Zero-banner interface; feedback is inline and color-coded.

### 🧠 Core Intelligence
- **Decision Engine**: Mistral-7b for routing and threat assessment.
- **Content Engine**: Llama-3.1 for secure response generation.
- **Credible Source Finder**: Python-based research agent with web access.

---

## 🛠️ Tech Stack

### Backend
- **Python 3.10+**
- **FastAPI**: High-performance API framework.
- **PyTorch & Transformers**: For DeBERTa ML model execution.
- **Ollama**: Local inference server for Mistral/Llama models.
- **LangChain**: Orchestration framework.

### Frontend
- **HTML5 / CSS3 / Vanilla JS**: Lightweight, dependency-free UI.
- **Cyberpunk Aesthetics**: Custom dark mode with neon accents.
- **2-Layer Input System**: Invisible textarea + overlay div for pixel-perfect syntax highlighting.

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- 8GB+ RAM (for local model inference)

### 1. Clone & Setup
```bash
git clone https://github.com/jayavardhanashoka/trinetra.git
cd trinetra
```

### 2. Install Dependencies
```bash
pip install fastapi uvicorn pydantic langchain-ollama requests beautifulsoup4 duckduckgo-search torch transformers
```

### 3. Pull AI Models
```bash
# Core LLMs
ollama pull llama3.1
ollama pull mistral

# Note: DeBERTa model will auto-download on first run
```

---

## 🚀 Usage

### 1. Start Ollama Server
```bash
ollama serve
```

### 2. Start Trinetra Backend
```bash
python main.py
```
*Server runs on port 8000. Look for `[STARTUP] Loading DeBERTa prompt injection model...` log.*

### 3. Launch Frontend
Open `index.html` in your browser.
*(Optional: Use a live server extension or `python -m http.server 8080`)*

---

## �️ Security Architecture

### Risk Scoring System
Trinetra calculates a total `risk_score` for every input:

| Signal | Weight | Description |
|--------|--------|-------------|
| **Override** | +10 | Instructions to ignore previous rules |
| **Bypass** | +10 | "Uncensored", "no limits" modes |
| **Extraction** | +10 | Attempts to leak system prompts |
| **Malicious** | +8 | Hostile intent validation |
| **ML (High)** | +8 | DeBERTa confidence >= 0.85 |
| **Roleplay** | +5 | "Act as...", "Simulate..." |
| **ML (Mod)** | +4 | DeBERTa confidence >= 0.70 |

**Decision Thresholds:**
- **WAIT (Blocked)**: Score ≥ 10
- **WARN (Restricted)**: Score ≥ 5
- **SAFE**: Score < 5

---

## 📡 API Reference

### 1. Real-time Check
`GET /realtime_detect?text=...`
- **Use**: Instant UI feedback (highlighting).
- **Return**: `{ matches: [...], status: "SAFE"|"WARNING"|"BLOCK" }`

### 2. Full Scan
`POST /scan`
- **Use**: Final submission and execution.
- **Payload**: `{"payload": "user input"}`
- **Return**: Full analysis, blocking verdict, or LLM response.

---

## 📄 License
© 2026 TRINETRA™. Protected under MIT License.
