"""
Unit tests for MedGemma Engine (demo mode — no GPU required).
Run: pytest tests/ -v
"""

import os
import sys
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

os.environ["DEMO_MODE"] = "true"  # Force demo mode for tests

import pytest
from PIL import Image

from medgemma_engine import (
    DEMO_RESPONSES,
    InferenceResult,
    MedGemmaEngine,
    get_engine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> MedGemmaEngine:
    """Return a fresh demo-mode engine."""
    e = MedGemmaEngine()
    assert e.is_demo, "Engine should be in demo mode for tests"
    return e


@pytest.fixture
def blank_image() -> Image.Image:
    """Return a minimal valid image."""
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


# ---------------------------------------------------------------------------
# Engine initialization
# ---------------------------------------------------------------------------


def test_engine_singleton():
    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2, "get_engine() must return the same singleton"


def test_engine_demo_mode(engine):
    assert engine.is_demo is True


def test_engine_device(engine):
    assert engine.device in ("cuda", "cpu")


def test_engine_loaded_model_is_none_in_demo(engine):
    assert engine.loaded_model is None


# ---------------------------------------------------------------------------
# Image analysis (demo mode)
# ---------------------------------------------------------------------------


def test_analyze_xray_demo(engine, blank_image):
    result = engine.analyze_medical_image(blank_image, "Describe this X-ray", image_type="xray")
    assert isinstance(result, InferenceResult)
    assert result.demo_mode is True
    assert len(result.text) > 50
    assert result.latency_ms >= 0


def test_analyze_dermatology_demo(engine, blank_image):
    result = engine.analyze_medical_image(
        blank_image, "Analyze this lesion", image_type="dermatology"
    )
    assert result.demo_mode is True
    assert "Dermatology" in result.text or "ABCDE" in result.text


def test_analyze_pathology_demo(engine, blank_image):
    result = engine.analyze_medical_image(
        blank_image, "Analyze slide", image_type="pathology"
    )
    assert result.demo_mode is True
    assert len(result.text) > 30


def test_analyze_unknown_image_type_falls_back(engine, blank_image):
    result = engine.analyze_medical_image(
        blank_image, "Analyze", image_type="unknown_type"
    )
    # Should fall back to xray demo response
    assert result.demo_mode is True
    assert len(result.text) > 0


# ---------------------------------------------------------------------------
# Medical Q&A (demo mode)
# ---------------------------------------------------------------------------


def test_medical_qa_demo(engine):
    result = engine.medical_qa("What are the symptoms of Type 2 Diabetes?")
    assert isinstance(result, InferenceResult)
    assert result.demo_mode is True
    assert len(result.text) > 30


def test_medical_qa_empty_question(engine):
    # Should still return a demo response, not crash
    result = engine.medical_qa("")
    assert result.demo_mode is True


# ---------------------------------------------------------------------------
# Clinical note summarization (demo mode)
# ---------------------------------------------------------------------------


def test_summarize_note_demo(engine):
    note = "Patient: 45M. CC: Chest pain. PMH: HTN. Assessment: Rule out ACS. Plan: EKG, troponin."
    result = engine.summarize_clinical_note(note)
    assert isinstance(result, InferenceResult)
    assert result.demo_mode is True
    assert len(result.text) > 30


# ---------------------------------------------------------------------------
# Triage (demo mode)
# ---------------------------------------------------------------------------


def test_triage_demo(engine):
    result = engine.triage_symptoms(
        symptoms="Severe chest pain radiating to left arm",
        patient_info="55yo male, HTN",
    )
    assert isinstance(result, InferenceResult)
    assert result.demo_mode is True
    assert len(result.text) > 30


# ---------------------------------------------------------------------------
# Demo responses coverage
# ---------------------------------------------------------------------------


def test_all_demo_responses_are_non_empty():
    for key, val in DEMO_RESPONSES.items():
        assert len(val) > 50, f"Demo response for '{key}' is too short"


# ---------------------------------------------------------------------------
# InferenceResult dataclass
# ---------------------------------------------------------------------------


def test_inference_result_creation():
    r = InferenceResult(
        text="Test result",
        model_id="test-model",
        latency_ms=123.4,
        demo_mode=True,
    )
    assert r.text == "Test result"
    assert r.model_id == "test-model"
    assert r.latency_ms == 123.4
    assert r.demo_mode is True
    assert r.error is None
