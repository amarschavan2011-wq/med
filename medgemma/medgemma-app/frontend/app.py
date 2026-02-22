"""
MedGemma Clinical AI — Streamlit Frontend

Modules:
  1. Medical Imaging (X-ray, Dermatology, Pathology)
  2. Medical Q&A / Knowledge Base
  3. Clinical Note Summarizer
  4. Patient Triage Assistant
  5. Dashboard & System Status
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — allow imports from backend when running standalone
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
PAGE_TITLE = "MedGemma Clinical AI"
PAGE_ICON = "🏥"

MODULES = {
    "Dashboard": "📊",
    "Medical Imaging": "🩻",
    "Medical Q&A": "💬",
    "Clinical Note Summarizer": "📋",
    "Patient Triage": "🚑",
    "About": "ℹ️",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://developers.google.com/health-ai-developer-foundations/medgemma",
        "Report a bug": "https://github.com/Google-Health/medgemma/issues",
        "About": "MedGemma Clinical AI — MedGemma Impact Challenge Submission",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 1rem; }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e8eaf6;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1a73e8; }
    .metric-card .label { font-size: 0.85rem; color: #555; margin-top: 0.2rem; }

    /* Result box */
    .result-box {
        background: #f8f9ff;
        border-left: 4px solid #1a73e8;
        border-radius: 0 8px 8px 0;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Warning */
    .medical-disclaimer {
        background: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #e65100;
        margin-bottom: 1rem;
    }

    /* Demo badge */
    .demo-badge {
        display: inline-block;
        background: #ff5722;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .live-badge {
        display: inline-block;
        background: #4caf50;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] > div:first-child {
        background: #0d47a1;
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] .stRadio label { color: white !important; }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
for key, default in {
    "api_available": False,
    "model_loaded": False,
    "demo_mode": True,
    "analysis_count": 0,
    "model_id": "Not loaded",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api_get(endpoint: str, timeout: int = 5) -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _api_post_json(endpoint: str, payload: dict, timeout: int = 60) -> Optional[dict]:
    try:
        r = requests.post(
            f"{API_BASE}{endpoint}", json=payload, timeout=timeout
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _api_post_file(
    endpoint: str,
    file_bytes: bytes,
    filename: str,
    extra_fields: dict | None = None,
    timeout: int = 120,
) -> Optional[dict]:
    try:
        files = {"file": (filename, file_bytes, "image/jpeg")}
        data = extra_fields or {}
        r = requests.post(
            f"{API_BASE}{endpoint}", files=files, data=data, timeout=timeout
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def refresh_status():
    """Poll /health and update session state."""
    health = _api_get("/health")
    if health:
        st.session_state.api_available = True
        st.session_state.model_loaded = health.get("model_loaded", False)
        st.session_state.demo_mode = health.get("demo_mode", True)
        st.session_state.model_id = health.get("model_id") or "Demo mode"
    else:
        st.session_state.api_available = False
        st.session_state.demo_mode = True


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<h2 style='color:white;text-align:center;margin-bottom:0.5rem;'>"
            "🏥 MedGemma AI</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#90caf9;text-align:center;font-size:0.8rem;"
            "margin-bottom:1.5rem;'>Clinical AI Platform</p>",
            unsafe_allow_html=True,
        )

        # Navigation
        page = st.radio(
            "Navigation",
            list(MODULES.keys()),
            format_func=lambda x: f"{MODULES[x]}  {x}",
            label_visibility="collapsed",
        )

        st.divider()

        # Status card
        refresh_status()
        api_ok = st.session_state.api_available
        demo = st.session_state.demo_mode

        st.markdown(
            f"""
            <div style="background:#1565c0;border-radius:8px;padding:0.8rem;">
                <b style="color:white;">System Status</b><br>
                <span style="color:{'#a5d6a7' if api_ok else '#ef9a9a'}">
                    {'✅' if api_ok else '❌'} API {'Online' if api_ok else 'Offline'}
                </span><br>
                <span style="color:{'#a5d6a7' if not demo else '#ffcc02'}">
                    {'✅ Model Live' if not demo else '⚡ Demo Mode'}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='color:#90caf9;font-size:0.72rem;margin-top:0.5rem;'>"
            f"Model: {st.session_state.model_id}</p>",
            unsafe_allow_html=True,
        )

        # Model loader
        st.divider()
        st.markdown(
            "<b style='color:white;font-size:0.9rem;'>Load Model</b>",
            unsafe_allow_html=True,
        )
        model_choice = st.selectbox(
            "Select model",
            [
                "google/medgemma-4b-it",
                "google/medgemma-1.5-4b-it",
                "google/medgemma-27b-text-it",
            ],
            label_visibility="collapsed",
        )
        if st.button("Load Model", use_container_width=True):
            with st.spinner("Loading model..."):
                resp = _api_post_json("/model/load", {"model_id": model_choice})
                if resp:
                    refresh_status()
                    if resp.get("success"):
                        st.success("Model loaded!")
                    else:
                        st.warning("Running in demo mode.")

        st.divider()
        st.markdown(
            "<p style='color:#90caf9;font-size:0.75rem;text-align:center;'>"
            "Powered by Google MedGemma<br>"
            "MedGemma Impact Challenge 2025<br>"
            "<a href='https://developers.google.com/health-ai-developer-foundations' "
            "style='color:#64b5f6;'>HAI-DEF</a></p>",
            unsafe_allow_html=True,
        )

    return page


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_dashboard():
    st.markdown(
        """
        <div class="main-header">
            <h1>🏥 MedGemma Clinical AI</h1>
            <p>Human-centered AI for healthcare — powered by Google MedGemma &amp; HAI-DEF</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="value">{st.session_state.analysis_count}</div>
                <div class="label">Analyses Run</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        status = "Online" if st.session_state.api_available else "Offline"
        st.markdown(
            f"""<div class="metric-card">
                <div class="value" style="font-size:1.2rem;">{status}</div>
                <div class="label">API Status</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        mode = "Demo" if st.session_state.demo_mode else "Live"
        st.markdown(
            f"""<div class="metric-card">
                <div class="value" style="font-size:1.2rem;">{mode}</div>
                <div class="label">Mode</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """<div class="metric-card">
                <div class="value">4</div>
                <div class="label">AI Modules</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("About This Application")
        st.markdown(
            """
This platform demonstrates how **Google MedGemma** and the **Health AI Developer Foundations (HAI-DEF)**
collection can power real-world clinical AI tools — without requiring centralized cloud infrastructure
or sending patient data to third-party APIs.

**Key Capabilities:**
- **Medical Image Analysis** — Chest X-ray, dermatology, histopathology interpretation
- **Medical Q&A** — Evidence-based answers to clinical questions using MedGemma 27B
- **Clinical Note Summarization** — Structured extraction from unstructured clinical text
- **Patient Triage** — AI-assisted symptom severity assessment

**Privacy-First Design:**
All model inference runs locally. Patient data never leaves your infrastructure.

**Models Used:**
| Model | Variant | Use Case |
|-------|---------|----------|
| `medgemma-4b-it` | Multimodal 4B | Image analysis |
| `medgemma-1.5-4b-it` | Multimodal 4B v1.5 | Advanced imaging |
| `medgemma-27b-text-it` | Text-only 27B | Medical Q&A |
            """
        )

    with col2:
        st.subheader("Quick Start")
        st.markdown(
            """
**Step 1:** Start the API backend
```bash
cd backend
python api.py
```

**Step 2:** (Optional) Load model
```
Use the sidebar → Load Model
```

**Step 3:** Choose a module
```
← Navigation on the left
```

**Step 4:** Upload an image or
ask a medical question.

---
> **Note:** The app runs in **Demo Mode** by default, showing representative AI responses without requiring a GPU.
            """
        )

    st.divider()
    st.subheader("Architecture Overview")
    st.markdown(
        """
```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ Imaging  │ │  Med QA  │ │  Notes   │ │    Triage    │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │
└───────┼────────────┼────────────┼───────────────┼──────────┘
        │            │            │               │
        └─────────────────────────────────────────┘
                              │ REST API (FastAPI)
              ┌───────────────▼────────────────┐
              │        FastAPI Backend          │
              │  /analyze/xray  /qa  /triage   │
              └───────────────┬────────────────┘
                              │
              ┌───────────────▼────────────────┐
              │      MedGemma Engine           │
              │  google/medgemma-4b-it         │
              │  google/medgemma-27b-text-it   │
              │  [Hugging Face Transformers]   │
              └────────────────────────────────┘
```
        """
    )


def page_medical_imaging():
    st.title("🩻 Medical Image Analysis")
    st.markdown(
        """<div class="medical-disclaimer">
        ⚠️ <b>Clinical Disclaimer:</b> This tool is for research and educational purposes only.
        AI-generated analyses are not a substitute for professional medical diagnosis.
        Always consult a qualified healthcare professional for clinical decisions.
        </div>""",
        unsafe_allow_html=True,
    )

    tab_xray, tab_derm, tab_path = st.tabs(
        ["🫁 Chest X-Ray", "🔬 Dermatology", "🧫 Histopathology"]
    )

    # -------- Chest X-Ray --------
    with tab_xray:
        st.subheader("Chest X-Ray Analysis")
        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded = st.file_uploader(
                "Upload chest X-ray (JPEG/PNG)",
                type=["jpg", "jpeg", "png"],
                key="xray_upload",
            )
            context = st.text_area(
                "Clinical context (optional)",
                placeholder="e.g., 45-year-old male with cough and fever for 3 days",
                height=80,
            )

            if uploaded:
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded X-Ray", use_container_width=True)

            analyze_btn = st.button(
                "Analyze X-Ray",
                type="primary",
                use_container_width=True,
                disabled=not uploaded,
            )

        with col2:
            if analyze_btn and uploaded:
                with st.spinner("Analyzing chest X-ray with MedGemma..."):
                    result = _api_post_file(
                        "/analyze/xray",
                        uploaded.getvalue(),
                        uploaded.name,
                        {"clinical_context": context},
                    )
                if result:
                    st.session_state.analysis_count += 1
                    badge = (
                        '<span class="demo-badge">DEMO</span>'
                        if result.get("demo_mode")
                        else '<span class="live-badge">LIVE</span>'
                    )
                    st.markdown(
                        f"**Radiology Report** {badge} "
                        f"<small>({result.get('latency_ms', 0):.0f} ms)</small>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="result-box">{result["result"].replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.download_button(
                        "Download Report",
                        result["result"],
                        file_name="xray_report.txt",
                        mime="text/plain",
                    )
            elif not uploaded:
                st.info("Upload a chest X-ray image to begin analysis.")
                st.markdown(
                    "**Sample use cases:**\n"
                    "- Pneumonia detection\n"
                    "- Cardiomegaly assessment\n"
                    "- Pleural effusion detection\n"
                    "- Pneumothorax identification"
                )

    # -------- Dermatology --------
    with tab_derm:
        st.subheader("Dermatology Image Analysis")
        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_derm = st.file_uploader(
                "Upload skin lesion image (JPEG/PNG)",
                type=["jpg", "jpeg", "png"],
                key="derm_upload",
            )
            history = st.text_area(
                "Lesion history (optional)",
                placeholder="e.g., Lesion present for 6 months, changing color",
                height=80,
            )
            if uploaded_derm:
                img = Image.open(uploaded_derm)
                st.image(img, caption="Skin Lesion", use_container_width=True)

            derm_btn = st.button(
                "Analyze Lesion",
                type="primary",
                use_container_width=True,
                disabled=not uploaded_derm,
            )

        with col2:
            if derm_btn and uploaded_derm:
                with st.spinner("Analyzing with MedGemma..."):
                    result = _api_post_file(
                        "/analyze/dermatology",
                        uploaded_derm.getvalue(),
                        uploaded_derm.name,
                        {"lesion_history": history},
                    )
                if result:
                    st.session_state.analysis_count += 1
                    badge = (
                        '<span class="demo-badge">DEMO</span>'
                        if result.get("demo_mode")
                        else '<span class="live-badge">LIVE</span>'
                    )
                    st.markdown(
                        f"**Dermatology Assessment** {badge}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="result-box">{result["result"].replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True,
                    )
            elif not uploaded_derm:
                st.info("Upload a skin lesion image.")
                st.markdown(
                    "**ABCDE Criteria Analysis:**\n"
                    "- **A**symmetry\n"
                    "- **B**order irregularity\n"
                    "- **C**olor variation\n"
                    "- **D**iameter\n"
                    "- **E**volution"
                )

    # -------- Histopathology --------
    with tab_path:
        st.subheader("Histopathology Slide Analysis")
        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_path = st.file_uploader(
                "Upload histopathology slide image",
                type=["jpg", "jpeg", "png"],
                key="path_upload",
            )
            stain = st.selectbox(
                "Stain type",
                ["H&E", "IHC", "PAS", "Masson's Trichrome", "Giemsa", "Other"],
            )
            clin_info = st.text_area(
                "Clinical information",
                placeholder="e.g., Breast biopsy, 58F, BRCA1+",
                height=80,
            )
            if uploaded_path:
                img = Image.open(uploaded_path)
                st.image(img, caption="Histopathology Slide", use_container_width=True)

            path_btn = st.button(
                "Analyze Slide",
                type="primary",
                use_container_width=True,
                disabled=not uploaded_path,
            )

        with col2:
            if path_btn and uploaded_path:
                with st.spinner("Analyzing histopathology with MedGemma..."):
                    result = _api_post_file(
                        "/analyze/pathology",
                        uploaded_path.getvalue(),
                        uploaded_path.name,
                        {"stain_type": stain, "clinical_info": clin_info},
                    )
                if result:
                    st.session_state.analysis_count += 1
                    st.markdown("**Pathology Report**")
                    st.markdown(
                        f'<div class="result-box">{result["result"].replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True,
                    )
            elif not uploaded_path:
                st.info("Upload a histopathology slide image.")


def page_medical_qa():
    st.title("💬 Medical Q&A")
    st.markdown(
        """<div class="medical-disclaimer">
        ⚠️ For educational and research purposes only. Not a substitute for professional medical advice.
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("Ask clinical questions powered by **MedGemma 27B** (87.7% on MedQA benchmark).")

    # Sample questions
    SAMPLE_QS = [
        "What are the diagnostic criteria for Type 2 Diabetes Mellitus?",
        "Explain the pathophysiology of COPD and key management strategies.",
        "What is the recommended treatment algorithm for Stage 3 CKD?",
        "Describe the clinical presentation and management of acute pulmonary embolism.",
        "What are the indications and contraindications for thrombolytic therapy?",
    ]

    col1, col2 = st.columns([2, 1])

    with col1:
        question = st.text_area(
            "Medical Question",
            placeholder="Enter your clinical question here...",
            height=120,
        )
        context = st.text_area(
            "Clinical Context (optional)",
            placeholder="Patient demographics, relevant history, comorbidities...",
            height=80,
        )

        submit = st.button("Get Answer", type="primary", use_container_width=True)

        if submit and question:
            with st.spinner("MedGemma is analyzing..."):
                result = _api_post_json(
                    "/qa", {"question": question, "context": context or None}
                )
            if result:
                st.session_state.analysis_count += 1
                badge = (
                    '<span class="demo-badge">DEMO</span>'
                    if result.get("demo_mode")
                    else '<span class="live-badge">LIVE</span>'
                )
                st.markdown(
                    f"**AI Response** {badge} "
                    f"<small>({result.get('latency_ms', 0):.0f} ms | {result.get('model_id', '')})</small>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="result-box">{result["result"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "Download Response",
                    f"Q: {question}\n\nA: {result['result']}",
                    file_name="medical_qa_response.txt",
                )

    with col2:
        st.markdown("**Sample Questions**")
        for i, q in enumerate(SAMPLE_QS):
            if st.button(q[:55] + "…" if len(q) > 55 else q, key=f"sample_{i}"):
                st.session_state["qa_prefill"] = q

        st.divider()
        st.markdown(
            """
**Model Performance:**
- MedQA: **87.7%** (27B)
- MedMCQA: **73.4%**
- PubMedQA: **79.3%**
- USMLE Step 1-3: Passing grade

*Source: Google Research, 2025*
            """
        )


def page_clinical_notes():
    st.title("📋 Clinical Note Summarizer")
    st.markdown(
        """<div class="medical-disclaimer">
        ⚠️ De-identify all patient information before processing. For research use only.
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "Paste an unstructured clinical note and MedGemma will extract a structured summary."
    )

    SAMPLE_NOTE = """
Date: 2024-01-15
Attending: Dr. Smith

Chief Complaint: 68-year-old male with 3-day history of progressive shortness of breath and productive cough.

History of Present Illness: Patient presents with worsening dyspnea on exertion and productive cough with yellowish sputum. Fever up to 38.8°C at home. No chest pain, no hemoptysis. Patient is a former smoker (30 pack-years, quit 5 years ago). PMH includes COPD (moderate, FEV1 62%), hypertension, and type 2 diabetes (HbA1c 7.2% last month).

Medications: Tiotropium 18mcg inhaler QD, Albuterol PRN, Lisinopril 10mg QD, Metformin 1000mg BID.

Vital Signs: BP 142/88 mmHg, HR 98 bpm, RR 22/min, SpO2 91% on room air, Temp 38.6°C.

Physical Exam: Mild respiratory distress. Decreased air entry at right base. Dullness to percussion right lower lobe. Bronchial breath sounds with egophony at right base.

Labs: WBC 14.2 K/uL (elevated), CRP 85 mg/L, PCT 0.8 ng/mL.

Assessment: Community-acquired pneumonia (CAP), right lower lobe, moderate severity (PSI Class III). COPD exacerbation component likely.

Plan:
1. IV Ceftriaxone 1g Q24h + Azithromycin 500mg QD x 5 days
2. Supplemental O2 to maintain SpO2 >94%
3. Bronchodilators: Albuterol nebulization Q4h
4. Systemic corticosteroids: Prednisone 40mg PO QD x 5 days
5. Chest X-ray for baseline, repeat in 4-6 weeks to confirm resolution
6. Monitor blood glucose closely given steroid use
7. Advance diet as tolerated, maintain hydration
8. Follow-up in 1 week with PCP
    """.strip()

    col1, col2 = st.columns([1, 1])

    with col1:
        note = st.text_area(
            "Clinical Note",
            value="",
            placeholder="Paste clinical note here (de-identified)...",
            height=400,
        )
        use_sample = st.checkbox("Use sample note")
        if use_sample:
            note = SAMPLE_NOTE

        summarize_btn = st.button(
            "Summarize & Extract",
            type="primary",
            use_container_width=True,
            disabled=not note.strip(),
        )

    with col2:
        st.markdown("**Structured Summary**")

        if summarize_btn and note.strip():
            with st.spinner("Processing clinical note with MedGemma NLP..."):
                result = _api_post_json(
                    "/clinical-note", {"note_text": note}
                )
            if result:
                st.session_state.analysis_count += 1
                badge = (
                    '<span class="demo-badge">DEMO</span>'
                    if result.get("demo_mode")
                    else '<span class="live-badge">LIVE</span>'
                )
                st.markdown(f"**Output** {badge}", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="result-box">{result["result"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "Download Summary",
                    result["result"],
                    file_name="clinical_summary.txt",
                )
        else:
            st.info(
                "Paste a clinical note or check 'Use sample note' to see this feature in action."
            )
            st.markdown(
                "**What gets extracted:**\n"
                "- Chief Complaint\n"
                "- History & Examination\n"
                "- Assessment\n"
                "- Plan & Medications\n"
                "- Follow-up items"
            )


def page_patient_triage():
    st.title("🚑 Patient Triage Assistant")
    st.markdown(
        """<div class="medical-disclaimer">
        ⚠️ <b>CRITICAL DISCLAIMER:</b> This tool is for demonstration only. It is NOT
        for real-world triage decisions. Always call emergency services (911) for
        life-threatening emergencies.
        </div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Patient Information")

        age = st.slider("Age", 0, 120, 45)
        sex = st.selectbox("Sex", ["Male", "Female", "Other/Unknown"])
        weight = st.number_input("Weight (kg)", 0.0, 300.0, 70.0)

        st.subheader("Symptoms")
        symptoms = st.text_area(
            "Describe symptoms in detail",
            placeholder=(
                "e.g., Chest pain radiating to left arm, onset 30 minutes ago, "
                "associated with sweating and shortness of breath..."
            ),
            height=150,
        )

        st.subheader("Vital Signs (if available)")
        c1, c2 = st.columns(2)
        with c1:
            bp_sys = st.number_input("BP Systolic (mmHg)", 0, 300, 120)
            hr = st.number_input("Heart Rate (bpm)", 0, 300, 80)
            spo2 = st.number_input("SpO2 (%)", 0, 100, 98)
        with c2:
            bp_dia = st.number_input("BP Diastolic (mmHg)", 0, 200, 80)
            rr = st.number_input("Resp Rate (/min)", 0, 60, 16)
            temp = st.number_input("Temperature (°C)", 30.0, 45.0, 37.0)

        comorbidities = st.multiselect(
            "Known conditions",
            [
                "Diabetes", "Hypertension", "Heart disease", "COPD/Asthma",
                "CKD", "Immunocompromised", "Pregnancy", "None",
            ],
            default=["None"],
        )

        triage_btn = st.button(
            "Assess Triage",
            type="primary",
            use_container_width=True,
            disabled=not symptoms.strip(),
        )

    with col2:
        st.subheader("Triage Assessment")

        if triage_btn and symptoms.strip():
            patient_info = (
                f"{age}-year-old {sex}, {weight}kg. "
                f"Conditions: {', '.join(comorbidities)}. "
                f"Vitals: BP {bp_sys}/{bp_dia} mmHg, HR {hr} bpm, "
                f"RR {rr}/min, SpO2 {spo2}%, Temp {temp}°C."
            )

            with st.spinner("Performing triage assessment..."):
                result = _api_post_json(
                    "/triage",
                    {"symptoms": symptoms, "patient_info": patient_info},
                )

            if result:
                st.session_state.analysis_count += 1
                badge = (
                    '<span class="demo-badge">DEMO</span>'
                    if result.get("demo_mode")
                    else '<span class="live-badge">LIVE</span>'
                )
                st.markdown(f"**Assessment Result** {badge}", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="result-box">{result["result"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )

                # Always show emergency reminder
                st.error(
                    "**REMINDER:** If you suspect a life-threatening emergency, "
                    "call 911 immediately. Do not rely on AI for emergencies."
                )

        else:
            # Acuity level reference
            st.markdown("**Emergency Triage Scale (ETS)**")
            levels = [
                ("🔴 Immediate", "Life-threatening — resuscitation required"),
                ("🟠 Emergent", "High risk — seen within 15 minutes"),
                ("🟡 Urgent", "Moderate risk — seen within 30 minutes"),
                ("🟢 Semi-urgent", "Low risk — seen within 1 hour"),
                ("⚪ Non-urgent", "Minimal — seen within 2 hours"),
            ]
            for level, desc in levels:
                st.markdown(f"**{level}:** {desc}")

            st.divider()
            st.info("Fill in patient details and symptoms to get a triage assessment.")


def page_about():
    st.title("ℹ️ About MedGemma Clinical AI")

    st.markdown(
        """
## Competition Submission

This application is a submission for the **[MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)**,
hosted by Google Research on Kaggle with a **$100,000 prize pool**.

---

## Models Used

### MedGemma 4B Multimodal (`google/medgemma-4b-it`)
- Built on Gemma 3 architecture
- SigLIP image encoder pre-trained on de-identified medical data
- Trained on: chest X-rays, dermatology, ophthalmology, histopathology
- Benchmark: 64.4% on MedQA | 81% radiologist-equivalent chest X-ray reports

### MedGemma 1.5 4B (`google/medgemma-1.5-4b-it`)
- Latest version (released January 2026)
- Improved: medical reasoning, records interpretation, high-dimensional imaging
- Supports: CT/MRI, longitudinal analysis, anatomical localization

### MedGemma 27B Text (`google/medgemma-27b-text-it`)
- Best-in-class small text model for medical Q&A
- Benchmark: 87.7% on MedQA (within 3 pts of DeepSeek R1)
- ~10x lower inference cost than equivalent models

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| AI Models | Google MedGemma (HAI-DEF) |
| Inference | HuggingFace Transformers 4.50+ |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| GPU Support | CUDA via PyTorch |

---

## Privacy & Ethics

- **Local inference** — patient data never leaves the server
- **Open weights** — full control over model and infrastructure
- **De-identification** — users are prompted to de-identify data
- **Disclaimers** — clear warnings that outputs are not clinical advice
- **Auditability** — all prompts and responses logged locally

---

## References

- [MedGemma — Google Developers](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [HAI-DEF Collection](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def)
- [MedGemma Research Blog](https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/)
- [MedGemma 1.5 Announcement](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)
- [Kaggle Competition](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
        """
    )


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def main():
    page = render_sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Medical Imaging":
        page_medical_imaging()
    elif page == "Medical Q&A":
        page_medical_qa()
    elif page == "Clinical Note Summarizer":
        page_clinical_notes()
    elif page == "Patient Triage":
        page_patient_triage()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
