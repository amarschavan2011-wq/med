# MedGemma Clinical AI Platform

> **MedGemma Impact Challenge Submission** — Human-centered AI for healthcare powered by Google MedGemma and HAI-DEF.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)

---

## Overview

This platform demonstrates how [Google MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma) and the [HAI-DEF collection](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def) can power real-world clinical AI tools that:

- Run **entirely on your own infrastructure** (no centralized cloud required)
- Keep **patient data private** (local inference only)
- Work across **multiple clinical domains** (imaging, Q&A, NLP, triage)

---

## Features

| Module | Description | Model |
|--------|-------------|-------|
| **Medical Imaging** | Chest X-ray, dermatology, histopathology AI analysis | MedGemma 4B / 1.5 4B |
| **Medical Q&A** | Evidence-based clinical question answering | MedGemma 27B Text |
| **Clinical Note NLP** | Structured extraction from unstructured notes | MedGemma 27B Text |
| **Patient Triage** | AI-assisted symptom severity assessment | MedGemma 4B |

---

## Quick Start

### Option 1: Standalone (Simplest)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run (Demo Mode — no GPU needed)
streamlit run standalone_app.py

# 3. Open browser: http://localhost:8501
```

### Option 2: Backend + Frontend (Recommended)

```bash
# Terminal 1: Start FastAPI backend
cd backend
DEMO_MODE=true python api.py

# Terminal 2: Start Streamlit frontend
cd frontend
API_BASE_URL=http://localhost:8000 streamlit run app.py
```

### Option 3: Docker Compose (Production)

```bash
# Demo mode (no GPU required)
docker-compose up --build

# With real model (requires GPU + HF token)
HF_TOKEN=hf_your_token DEMO_MODE=false docker-compose up --build
```

---

## Loading the Real MedGemma Model

1. Accept the model license on HuggingFace:
   - [medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
   - [medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
   - [medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it)

2. Get a HuggingFace token: https://huggingface.co/settings/tokens

3. Configure environment:

```bash
cp .env.example .env
# Edit .env:
#   HF_TOKEN=hf_your_token_here
#   DEMO_MODE=false
#   MEDGEMMA_MODEL=google/medgemma-4b-it
```

4. Start the application (GPU strongly recommended for 4B model, required for 27B).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Streamlit Frontend (:8501)              │
│  ┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐   │
│  │ Imaging  │ │ Med QA │ │ Notes  │ │ Triage   │   │
│  └────┬─────┘ └───┬────┘ └───┬────┘ └────┬─────┘   │
└───────┼───────────┼──────────┼───────────┼─────────┘
        └───────────────────────────────────┘
                          │ REST API
        ┌─────────────────▼──────────────────┐
        │       FastAPI Backend (:8000)       │
        │  /analyze/xray  /analyze/dermatology│
        │  /analyze/pathology  /qa  /triage  │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │         MedGemma Engine             │
        │  google/medgemma-4b-it             │
        │  google/medgemma-1.5-4b-it         │
        │  google/medgemma-27b-text-it       │
        │  [HuggingFace Transformers 4.50+]  │
        └────────────────────────────────────┘
```

---

## API Reference

Start the backend (`python backend/api.py`) and visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health + model status |
| POST | `/analyze/xray` | Chest X-ray analysis |
| POST | `/analyze/dermatology` | Skin lesion analysis |
| POST | `/analyze/pathology` | Histopathology slide analysis |
| POST | `/qa` | Medical question answering |
| POST | `/clinical-note` | Clinical note summarization |
| POST | `/triage` | Patient symptom triage |
| POST | `/model/load` | Load / switch model |

---

## Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## Hardware Requirements

| Model | Min GPU RAM | Notes |
|-------|-------------|-------|
| Demo mode | None | No GPU needed |
| MedGemma 4B | ~8 GB VRAM | NVIDIA T4 / RTX 3080 |
| MedGemma 1.5 4B | ~8 GB VRAM | NVIDIA T4 / RTX 3080 |
| MedGemma 27B | ~55 GB VRAM | A100 80GB recommended |

---

## Model Performance

| Benchmark | MedGemma 4B | MedGemma 27B |
|-----------|-------------|--------------|
| MedQA | 64.4% | **87.7%** |
| MedMCQA | ~58% | 73.4% |
| PubMedQA | ~72% | 79.3% |
| Chest X-ray equivalence | **81%** | — |

*Source: [Google Research, 2025](https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/)*

---

## Safety & Ethical Considerations

- All outputs are marked with clear disclaimers
- Application explicitly states it is **not for clinical diagnosis**
- Users are instructed to de-identify patient data
- Emergency triage module always redirects to professional care
- Local inference preserves patient privacy

---

## References

- [MedGemma — Google Developers](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [HAI-DEF HuggingFace Collection](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def)
- [MedGemma 1.5 Announcement](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)
- [MedGemma GitHub](https://github.com/Google-Health/medgemma)
- [Kaggle Competition](https://www.kaggle.com/competitions/med-gemma-impact-challenge)

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

> **Medical Disclaimer:** This software is intended for research and educational purposes only. It is not a medical device and should not be used for clinical diagnosis or treatment decisions. Always consult a qualified healthcare professional.
