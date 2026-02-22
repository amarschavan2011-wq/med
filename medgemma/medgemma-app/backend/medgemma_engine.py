"""
MedGemma Engine — Core model loading and inference module.

Supports:
  - google/medgemma-4b-it        (multimodal: image + text)
  - google/medgemma-1.5-4b-it    (multimodal: image + text, latest)
  - google/medgemma-27b-text-it  (text-only, large)

Falls back to a mock/demo mode when no GPU / HF token is available,
so the app can be demonstrated without a real model download.
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import requests
import torch
from loguru import logger
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")          # set in .env
DEFAULT_MODEL = os.getenv(
    "MEDGEMMA_MODEL", "google/medgemma-4b-it"
)
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))


class ModelVariant(str, Enum):
    MULTIMODAL_4B = "google/medgemma-4b-it"
    MULTIMODAL_4B_V15 = "google/medgemma-1.5-4b-it"
    TEXT_27B = "google/medgemma-27b-text-it"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    text: str
    model_id: str
    latency_ms: float
    demo_mode: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Demo responses (used when DEMO_MODE=true or model not loaded)
# ---------------------------------------------------------------------------

DEMO_RESPONSES: dict[str, str] = {
    "xray": (
        "**Chest X-Ray Analysis (Demo)**\n\n"
        "**Findings:**\n"
        "- Lung fields appear clear bilaterally with no focal consolidation or effusion detected.\n"
        "- Cardiac silhouette is within normal limits (cardiothoracic ratio < 0.5).\n"
        "- Mediastinum is not widened.\n"
        "- Bony structures intact — no acute rib fractures visualized.\n"
        "- Diaphragmatic contours are sharp and symmetric.\n\n"
        "**Impression:**\n"
        "No acute cardiopulmonary process identified. Findings are within normal limits for the patient's age group.\n\n"
        "*⚠️ This is a demonstration response. In production, MedGemma 4B/1.5 performs actual image analysis.*"
    ),
    "dermatology": (
        "**Dermatology Image Analysis (Demo)**\n\n"
        "**Observed Features:**\n"
        "- Lesion border: irregular with asymmetric margins.\n"
        "- Color variation: mixed pigmentation noted.\n"
        "- Diameter: estimated >6 mm based on reference.\n"
        "- Surface: slightly elevated, non-ulcerated.\n\n"
        "**Differential Diagnosis:**\n"
        "1. Dysplastic nevus — most likely\n"
        "2. Superficial spreading melanoma — cannot exclude\n"
        "3. Seborrheic keratosis — less likely given morphology\n\n"
        "**Recommendation:**\n"
        "Refer to dermatologist for dermoscopic evaluation and possible biopsy.\n\n"
        "*⚠️ Demo mode active — not a real clinical assessment.*"
    ),
    "pathology": (
        "**Histopathology Slide Analysis (Demo)**\n\n"
        "**Tissue Architecture:**\n"
        "- Glandular structures present with moderate nuclear pleomorphism.\n"
        "- Increased nuclear-to-cytoplasmic ratio observed.\n"
        "- Mitotic figures: 3–5 per high-power field.\n"
        "- Stromal invasion pattern consistent with invasive carcinoma.\n\n"
        "**Grading (Nottingham):**\n"
        "- Tubule formation: Score 3\n"
        "- Nuclear pleomorphism: Score 2\n"
        "- Mitotic count: Score 2\n"
        "- **Overall Grade: Grade 2 (moderately differentiated)**\n\n"
        "*⚠️ Demo mode — real analysis requires MedGemma model loaded.*"
    ),
    "medical_qa": (
        "**Medical Information (Demo)**\n\n"
        "Based on the clinical question, here is a structured evidence-based response:\n\n"
        "The condition described involves a complex interplay of physiological factors. "
        "Current clinical guidelines recommend a stepwise diagnostic approach, beginning "
        "with thorough history taking and physical examination, followed by targeted "
        "laboratory investigations and imaging as clinically indicated.\n\n"
        "Key considerations include patient comorbidities, medication history, and "
        "potential contraindications. Evidence-based management should follow current "
        "specialty society guidelines.\n\n"
        "*⚠️ Demo mode — MedGemma 27B provides detailed, benchmarked medical Q&A.*"
    ),
    "clinical_note": (
        "**Clinical Note Summary (Demo)**\n\n"
        "**Patient:** [De-identified]\n"
        "**Date:** [Encounter date]\n\n"
        "**Chief Complaint:** The patient presents with the symptoms described.\n\n"
        "**Assessment:**\n"
        "Based on clinical presentation, the differential includes primary condition "
        "with secondary contributing factors. Vital signs were within acceptable ranges.\n\n"
        "**Plan:**\n"
        "1. Order CBC, CMP, and targeted imaging.\n"
        "2. Prescribe symptomatic treatment as indicated.\n"
        "3. Follow-up in 2 weeks or sooner if symptoms worsen.\n"
        "4. Patient education provided regarding warning signs.\n\n"
        "*⚠️ Demo mode — real note analysis uses MedGemma NLP pipeline.*"
    ),
    "patient_triage": (
        "**Symptom Triage Assessment (Demo)**\n\n"
        "**Acuity Level: MODERATE (Orange)**\n\n"
        "**Symptom Analysis:**\n"
        "The described symptoms suggest a condition requiring prompt evaluation within "
        "2–4 hours. No immediate life-threatening indicators identified based on "
        "presented information.\n\n"
        "**Red Flag Signs to Monitor:**\n"
        "- Sudden severe chest pain or shortness of breath → Emergency (911)\n"
        "- Altered consciousness or confusion → Emergency (911)\n"
        "- High fever >103°F unresponsive to antipyretics → Urgent care\n\n"
        "**Recommended Action:**\n"
        "Visit urgent care or contact your primary care physician today.\n\n"
        "*⚠️ Demo mode. NOT for real triage decisions.*"
    ),
}


# ---------------------------------------------------------------------------
# Model Manager (singleton pattern)
# ---------------------------------------------------------------------------

class MedGemmaEngine:
    """
    Lazily loads MedGemma models and manages inference.
    Automatically falls back to demo mode if loading fails.
    """

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._current_model_id: Optional[str] = None
        self._demo_mode: bool = DEMO_MODE
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"MedGemmaEngine init | device={self._device} | demo_mode={self._demo_mode}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_demo(self) -> bool:
        return self._demo_mode

    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded_model(self) -> Optional[str]:
        return self._current_model_id

    def load_model(self, model_id: str = DEFAULT_MODEL) -> bool:
        """Load a MedGemma model. Returns True on success."""
        if self._demo_mode:
            logger.info("Demo mode active — skipping model load.")
            return False

        if model_id == self._current_model_id and self._model is not None:
            logger.info(f"Model {model_id} already loaded.")
            return True

        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            logger.info(f"Loading model: {model_id}")
            token = HF_TOKEN or None

            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if self._device == "cuda" else None,
                token=token,
                low_cpu_mem_usage=True,
            )
            self._processor = AutoProcessor.from_pretrained(
                model_id, token=token
            )
            self._current_model_id = model_id
            self._demo_mode = False
            logger.success(f"Model loaded: {model_id}")
            return True

        except Exception as exc:
            logger.error(f"Failed to load model {model_id}: {exc}")
            self._demo_mode = True
            return False

    def analyze_medical_image(
        self,
        image: Image.Image,
        prompt: str,
        image_type: str = "xray",
    ) -> InferenceResult:
        """Run image + text inference (MedGemma 4B multimodal)."""
        t0 = time.perf_counter()

        if self._demo_mode or self._model is None:
            demo_text = DEMO_RESPONSES.get(image_type, DEMO_RESPONSES["xray"])
            return InferenceResult(
                text=demo_text,
                model_id="demo-mode",
                latency_ms=(time.perf_counter() - t0) * 1000,
                demo_mode=True,
            )

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
            inputs = {
                k: v.to(self._model.device, dtype=dtype)
                if v.dtype in (torch.float32, torch.float16, torch.bfloat16)
                else v.to(self._model.device)
                for k, v in inputs.items()
            }

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self._model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
                generation = generation[0][input_len:]

            decoded = self._processor.decode(generation, skip_special_tokens=True)
            latency = (time.perf_counter() - t0) * 1000
            return InferenceResult(
                text=decoded,
                model_id=self._current_model_id,
                latency_ms=latency,
            )

        except Exception as exc:
            logger.error(f"Inference error: {exc}")
            return InferenceResult(
                text=DEMO_RESPONSES.get(image_type, ""),
                model_id=self._current_model_id or "unknown",
                latency_ms=(time.perf_counter() - t0) * 1000,
                demo_mode=True,
                error=str(exc),
            )

    def medical_qa(self, question: str) -> InferenceResult:
        """Run text-only medical Q&A."""
        t0 = time.perf_counter()

        if self._demo_mode or self._model is None:
            return InferenceResult(
                text=DEMO_RESPONSES["medical_qa"],
                model_id="demo-mode",
                latency_ms=(time.perf_counter() - t0) * 1000,
                demo_mode=True,
            )

        try:
            messages = [{"role": "user", "content": question}]

            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self._model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
                generation = generation[0][input_len:]

            decoded = self._processor.decode(generation, skip_special_tokens=True)
            return InferenceResult(
                text=decoded,
                model_id=self._current_model_id,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        except Exception as exc:
            logger.error(f"QA inference error: {exc}")
            return InferenceResult(
                text=DEMO_RESPONSES["medical_qa"],
                model_id=self._current_model_id or "unknown",
                latency_ms=(time.perf_counter() - t0) * 1000,
                demo_mode=True,
                error=str(exc),
            )

    def summarize_clinical_note(self, note_text: str) -> InferenceResult:
        """Summarize / extract structured data from a clinical note."""
        prompt = (
            "You are a clinical NLP assistant. Analyze the following clinical note and "
            "produce a structured summary with sections: Chief Complaint, History, "
            "Physical Exam findings, Assessment, Plan, and Key Follow-up items.\n\n"
            f"CLINICAL NOTE:\n{note_text}\n\nSTRUCTURED SUMMARY:"
        )
        result = self.medical_qa(prompt)
        if result.demo_mode:
            result.text = DEMO_RESPONSES["clinical_note"]
        return result

    def triage_symptoms(self, symptoms: str, patient_info: str) -> InferenceResult:
        """AI-powered symptom triage."""
        prompt = (
            "You are a clinical decision support tool. Based on the patient information "
            "and symptoms described, provide a triage acuity level (Immediate/Emergent/"
            "Urgent/Semi-urgent/Non-urgent), key red-flag signs, and recommended "
            "next steps. Always recommend professional medical consultation.\n\n"
            f"PATIENT INFO: {patient_info}\n"
            f"SYMPTOMS: {symptoms}\n\n"
            "TRIAGE ASSESSMENT:"
        )
        result = self.medical_qa(prompt)
        if result.demo_mode:
            result.text = DEMO_RESPONSES["patient_triage"]
        return result


# Module-level singleton
_engine: Optional[MedGemmaEngine] = None


def get_engine() -> MedGemmaEngine:
    global _engine
    if _engine is None:
        _engine = MedGemmaEngine()
    return _engine
