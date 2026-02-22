"""
Integration tests for the FastAPI backend (demo mode).
Run: pytest tests/ -v
"""

import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
os.environ["DEMO_MODE"] = "true"

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Import the app — must be done after setting env vars
from api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_image_bytes(width: int = 128, height: int = 128) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 120, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Health & System
# ---------------------------------------------------------------------------


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "demo_mode" in data
    assert "device" in data


def test_model_status_endpoint():
    resp = client.get("/model/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "demo_mode" in data


def test_load_model_demo_mode():
    resp = client.post(
        "/model/load",
        json={"model_id": "google/medgemma-4b-it"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "demo_mode" in data
    assert "message" in data


# ---------------------------------------------------------------------------
# Medical Imaging
# ---------------------------------------------------------------------------


def test_analyze_xray():
    img_bytes = _make_image_bytes()
    resp = client.post(
        "/analyze/xray",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"clinical_context": "45yo male, cough"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    assert len(data["result"]) > 10
    assert "latency_ms" in data
    assert "demo_mode" in data


def test_analyze_dermatology():
    img_bytes = _make_image_bytes()
    resp = client.post(
        "/analyze/dermatology",
        files={"file": ("lesion.jpg", img_bytes, "image/jpeg")},
        data={"lesion_history": "6 months"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    assert len(data["result"]) > 10


def test_analyze_pathology():
    img_bytes = _make_image_bytes()
    resp = client.post(
        "/analyze/pathology",
        files={"file": ("slide.jpg", img_bytes, "image/jpeg")},
        data={"stain_type": "H&E", "clinical_info": "Breast biopsy"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data


def test_analyze_image_generic():
    img_bytes = _make_image_bytes()
    resp = client.post(
        "/analyze/image",
        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
        data={"prompt": "Describe this image", "image_type": "xray"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data


def test_analyze_xray_no_file():
    resp = client.post("/analyze/xray")
    assert resp.status_code == 422  # Validation error


def test_analyze_xray_invalid_file():
    resp = client.post(
        "/analyze/xray",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Medical Q&A
# ---------------------------------------------------------------------------


def test_medical_qa():
    resp = client.post("/qa", json={"question": "What are symptoms of pneumonia?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    assert len(data["result"]) > 10


def test_medical_qa_with_context():
    resp = client.post(
        "/qa",
        json={
            "question": "Should this patient be admitted?",
            "context": "68yo male with pneumonia, SpO2 91%",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data


def test_medical_qa_empty_question():
    resp = client.post("/qa", json={"question": "ab"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Clinical Note
# ---------------------------------------------------------------------------


def test_clinical_note():
    resp = client.post(
        "/clinical-note",
        json={"note_text": "Patient presents with chest pain. BP 160/100. Plan: EKG."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data


def test_clinical_note_too_short():
    resp = client.post("/clinical-note", json={"note_text": "hi"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------


def test_triage():
    resp = client.post(
        "/triage",
        json={
            "symptoms": "Severe chest pain with diaphoresis, onset 20 min ago",
            "patient_info": "55yo male, diabetic",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    assert len(data["result"]) > 10


def test_triage_default_patient_info():
    resp = client.post(
        "/triage",
        json={"symptoms": "Headache for 2 days"},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------


def test_openapi_docs():
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_openapi_schema():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    data = resp.json()
    assert "paths" in data
