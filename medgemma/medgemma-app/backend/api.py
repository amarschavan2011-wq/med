"""
MedGemma Clinical AI — FastAPI Backend

Endpoints:
  GET  /health              — health-check + model status
  POST /analyze/image       — medical image analysis (multimodal)
  POST /analyze/xray        — chest X-ray specific analysis
  POST /analyze/dermatology — dermatology image analysis
  POST /analyze/pathology   — histopathology analysis
  POST /qa                  — medical question answering
  POST /clinical-note       — clinical note summarization
  POST /triage              — patient symptom triage
  POST /model/load          — load / switch model
  GET  /model/status        — current model info
"""

from __future__ import annotations

import base64
import io
import os
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from medgemma_engine import (
    DEMO_RESPONSES,
    InferenceResult,
    ModelVariant,
    get_engine,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MedGemma Clinical AI API",
    description=(
        "A human-centered healthcare AI application powered by Google MedGemma "
        "and the HAI-DEF collection. Built for the MedGemma Impact Challenge."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class QARequest(BaseModel):
    question: str = Field(..., min_length=5, description="Medical question")
    context: Optional[str] = Field(None, description="Optional clinical context")


class ClinicalNoteRequest(BaseModel):
    note_text: str = Field(..., min_length=10, description="Full clinical note text")


class TriageRequest(BaseModel):
    symptoms: str = Field(..., description="Patient symptoms description")
    patient_info: str = Field(
        default="Unknown age, unknown sex",
        description="Basic patient demographics",
    )


class ModelLoadRequest(BaseModel):
    model_id: str = Field(
        default="google/medgemma-4b-it",
        description="HuggingFace model ID for MedGemma",
    )


class AnalysisResponse(BaseModel):
    result: str
    model_id: str
    latency_ms: float
    demo_mode: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: Optional[str]
    device: str
    demo_mode: bool
    cuda_available: bool


# ---------------------------------------------------------------------------
# Helper: parse uploaded image
# ---------------------------------------------------------------------------


def _load_image(file: UploadFile) -> Image.Image:
    contents = file.file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        return img
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")


def _result_to_response(result: InferenceResult) -> AnalysisResponse:
    return AnalysisResponse(
        result=result.text,
        model_id=result.model_id,
        latency_ms=round(result.latency_ms, 2),
        demo_mode=result.demo_mode,
        error=result.error,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    engine = get_engine()
    return HealthResponse(
        status="healthy",
        model_loaded=engine.loaded_model is not None,
        model_id=engine.loaded_model,
        device=engine.device,
        demo_mode=engine.is_demo,
        cuda_available=torch.cuda.is_available(),
    )


@app.post("/model/load", tags=["System"])
async def load_model(req: ModelLoadRequest):
    """Load or switch the active MedGemma model."""
    engine = get_engine()
    success = engine.load_model(req.model_id)
    return {
        "success": success,
        "model_id": req.model_id,
        "demo_mode": engine.is_demo,
        "message": (
            f"Model {req.model_id} loaded successfully."
            if success
            else "Running in demo mode (model not loaded)."
        ),
    }


@app.get("/model/status", tags=["System"])
async def model_status():
    engine = get_engine()
    return {
        "model_id": engine.loaded_model,
        "demo_mode": engine.is_demo,
        "device": engine.device,
    }


@app.post("/analyze/image", response_model=AnalysisResponse, tags=["Medical Imaging"])
async def analyze_image(
    file: UploadFile = File(..., description="Medical image (JPEG/PNG)"),
    prompt: str = Form(
        default="Analyze this medical image and describe your findings.",
        description="Analysis prompt",
    ),
    image_type: str = Form(default="xray", description="xray | dermatology | pathology"),
):
    """General medical image analysis using MedGemma multimodal."""
    engine = get_engine()
    image = _load_image(file)
    result = engine.analyze_medical_image(image, prompt, image_type=image_type)
    return _result_to_response(result)


@app.post("/analyze/xray", response_model=AnalysisResponse, tags=["Medical Imaging"])
async def analyze_xray(
    file: UploadFile = File(..., description="Chest X-ray image"),
    clinical_context: str = Form(
        default="",
        description="Optional clinical context (age, symptoms, prior history)",
    ),
):
    """Analyze a chest X-ray image using MedGemma."""
    engine = get_engine()
    image = _load_image(file)
    prompt = (
        "You are an expert radiologist. Carefully analyze this chest X-ray and provide:\n"
        "1. Systematic review of lung fields, heart, mediastinum, and bones\n"
        "2. Identification of any abnormalities\n"
        "3. Impression and recommendation\n"
        f"{'Clinical context: ' + clinical_context if clinical_context else ''}\n"
        "Provide a structured radiology report."
    )
    result = engine.analyze_medical_image(image, prompt, image_type="xray")
    return _result_to_response(result)


@app.post("/analyze/dermatology", response_model=AnalysisResponse, tags=["Medical Imaging"])
async def analyze_dermatology(
    file: UploadFile = File(..., description="Dermatology image"),
    lesion_history: str = Form(default="", description="Duration and history of lesion"),
):
    """Analyze a dermatology image using MedGemma."""
    engine = get_engine()
    image = _load_image(file)
    prompt = (
        "You are a board-certified dermatologist. Analyze this skin lesion image and provide:\n"
        "1. ABCDE criteria assessment (Asymmetry, Border, Color, Diameter, Evolution)\n"
        "2. Morphological description\n"
        "3. Differential diagnosis (top 3)\n"
        "4. Management recommendation\n"
        f"{'Patient history: ' + lesion_history if lesion_history else ''}"
    )
    result = engine.analyze_medical_image(image, prompt, image_type="dermatology")
    return _result_to_response(result)


@app.post("/analyze/pathology", response_model=AnalysisResponse, tags=["Medical Imaging"])
async def analyze_pathology(
    file: UploadFile = File(..., description="Histopathology slide image"),
    stain_type: str = Form(default="H&E", description="Staining type (H&E, IHC, etc.)"),
    clinical_info: str = Form(default="", description="Clinical context"),
):
    """Analyze a histopathology slide using MedGemma."""
    engine = get_engine()
    image = _load_image(file)
    prompt = (
        f"You are an expert pathologist. Analyze this {stain_type}-stained histopathology "
        "slide and provide:\n"
        "1. Tissue architecture and cellular morphology\n"
        "2. Mitotic count (per HPF if visible)\n"
        "3. Histological grade if applicable\n"
        "4. Diagnosis and differential\n"
        f"{'Clinical context: ' + clinical_info if clinical_info else ''}"
    )
    result = engine.analyze_medical_image(image, prompt, image_type="pathology")
    return _result_to_response(result)


@app.post("/qa", response_model=AnalysisResponse, tags=["Medical Q&A"])
async def medical_qa(req: QARequest):
    """Answer medical questions using MedGemma."""
    engine = get_engine()
    full_question = req.question
    if req.context:
        full_question = f"Context: {req.context}\n\nQuestion: {req.question}"
    result = engine.medical_qa(full_question)
    return _result_to_response(result)


@app.post("/clinical-note", response_model=AnalysisResponse, tags=["Clinical NLP"])
async def summarize_clinical_note(req: ClinicalNoteRequest):
    """Summarize and extract structured data from a clinical note."""
    engine = get_engine()
    result = engine.summarize_clinical_note(req.note_text)
    return _result_to_response(result)


@app.post("/triage", response_model=AnalysisResponse, tags=["Patient Care"])
async def triage_patient(req: TriageRequest):
    """AI-powered patient symptom triage (decision support only)."""
    engine = get_engine()
    result = engine.triage_symptoms(req.symptoms, req.patient_info)
    return _result_to_response(result)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("ENV", "production") == "development",
        log_level="info",
    )
