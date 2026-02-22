"""
MedGemma Clinical AI — STANDALONE Single-File Application

Run directly without the separate FastAPI backend:
    pip install -r requirements.txt
    streamlit run standalone_app.py

This file bundles the entire application (engine + UI) into one script.
Perfect for Kaggle notebook submissions and quick demos.
"""

from __future__ import annotations

import io
import os
import sys
import time
from typing import Optional

import streamlit as st
from PIL import Image

# ============================================================
# Inline MedGemma Engine (no external import needed)
# ============================================================

import torch

HF_TOKEN = os.getenv("HF_TOKEN", "")
DEFAULT_MODEL = os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "400"))

DEMO_RESPONSES = {
    "xray": (
        "**Chest X-Ray Analysis**\n\n"
        "**Systematic Review:**\n"
        "- **Lung fields:** No focal consolidation, air-space opacity, or pleural effusion bilaterally.\n"
        "- **Heart:** Cardiothoracic ratio within normal limits (~0.45). No cardiomegaly.\n"
        "- **Mediastinum:** Not widened. No shift. Trachea midline.\n"
        "- **Hila:** Not enlarged bilaterally.\n"
        "- **Diaphragm:** Visualized bilaterally with sharp costophrenic angles.\n"
        "- **Bony structures:** No acute rib fractures, clavicles and scapulae intact.\n\n"
        "**Impression:**\n"
        "No acute cardiopulmonary abnormality detected on this PA chest radiograph. "
        "Findings are within normal limits.\n\n"
        "*⚡ Demo Mode — Load MedGemma model for real AI analysis.*"
    ),
    "dermatology": (
        "**Dermatology Assessment**\n\n"
        "**ABCDE Criteria:**\n"
        "- **A (Asymmetry):** Mild asymmetry noted in lesion contour.\n"
        "- **B (Border):** Slightly irregular border with indistinct edges at 3 o'clock.\n"
        "- **C (Color):** Heterogeneous pigmentation — brown with darker focal areas.\n"
        "- **D (Diameter):** Estimated ~8mm based on image scale.\n"
        "- **E (Evolution):** Not assessable from single image.\n\n"
        "**Differential Diagnosis (ranked):**\n"
        "1. Dysplastic (atypical) nevus — most likely\n"
        "2. Superficial spreading melanoma — cannot exclude\n"
        "3. Pigmented basal cell carcinoma — less likely\n\n"
        "**Recommendation:** Dermoscopic evaluation by dermatologist. "
        "Consider excisional biopsy if clinical suspicion high.\n\n"
        "*⚡ Demo Mode — Load MedGemma model for real AI analysis.*"
    ),
    "pathology": (
        "**Histopathology Report**\n\n"
        "**Morphological Findings:**\n"
        "- Disrupted glandular architecture with cribriform pattern.\n"
        "- Nuclear enlargement with prominent nucleoli.\n"
        "- Mitotic figures: 4/10 HPF.\n"
        "- Stromal desmoplasia present.\n"
        "- No lymphovascular invasion identified in this section.\n\n"
        "**Nottingham Grade:**\n"
        "- Tubule formation: 3 | Nuclear pleomorphism: 2 | Mitotic count: 2\n"
        "- **Total: 7 → Grade 2 (Moderately Differentiated)**\n\n"
        "**Diagnosis:** Invasive ductal carcinoma, NOS — Grade 2\n\n"
        "*⚡ Demo Mode — Load MedGemma model for real AI analysis.*"
    ),
    "medical_qa": (
        "**Medical Information**\n\n"
        "Based on current clinical evidence and guidelines, here is a structured response:\n\n"
        "The condition involves multifactorial pathophysiology with both genetic and "
        "environmental contributors. Diagnosis relies on clinical criteria combined with "
        "targeted investigations. Management follows a stepwise approach beginning with "
        "lifestyle modifications, progressing to pharmacotherapy as indicated.\n\n"
        "Key evidence-based recommendations include:\n"
        "1. Initial assessment with complete history and physical examination.\n"
        "2. Targeted laboratory workup (CBC, metabolic panel, relevant biomarkers).\n"
        "3. Imaging as clinically indicated.\n"
        "4. Pharmacotherapy per current specialty guidelines.\n"
        "5. Regular monitoring and adjustment of treatment plan.\n\n"
        "*⚡ Demo Mode — Load MedGemma 27B for detailed, benchmarked medical answers.*"
    ),
    "clinical_note": (
        "**Structured Clinical Summary**\n\n"
        "**Chief Complaint:** As described in the clinical note.\n\n"
        "**History of Present Illness:**\n"
        "Patient presents with the symptoms detailed, with relevant timeline and "
        "associated features noted. Past medical history includes relevant comorbidities.\n\n"
        "**Physical Examination:**\n"
        "Key findings include the abnormalities documented in the original note.\n\n"
        "**Assessment:**\n"
        "Primary diagnosis with secondary contributing factors. "
        "Severity classified as moderate based on clinical criteria.\n\n"
        "**Plan:**\n"
        "1. Investigations as ordered.\n"
        "2. Pharmacotherapy initiated per guidelines.\n"
        "3. Supportive care measures.\n"
        "4. Follow-up scheduled with appropriate parameters to monitor.\n\n"
        "**Key Follow-up Items:** Monitor response to treatment at next visit.\n\n"
        "*⚡ Demo Mode — Load MedGemma for real NLP extraction.*"
    ),
    "patient_triage": (
        "**Triage Assessment**\n\n"
        "**Acuity Level: URGENT (Yellow) — Seen within 30 minutes**\n\n"
        "**Clinical Reasoning:**\n"
        "The symptom constellation described warrants prompt evaluation. "
        "Vital sign parameters suggest hemodynamic stability, but symptom severity "
        "and potential underlying etiology require timely assessment.\n\n"
        "**Red Flag Signs — Seek immediate emergency care if:**\n"
        "- Sudden severe chest pain or pressure\n"
        "- Difficulty breathing at rest\n"
        "- Loss of consciousness or severe confusion\n"
        "- Signs of stroke (FAST: Face, Arms, Speech, Time)\n\n"
        "**Recommended Action:**\n"
        "Proceed to urgent care or emergency department. "
        "Contact primary care physician if symptoms are mild and stable.\n\n"
        "**⚠️ CRITICAL: This is NOT a substitute for real medical assessment. "
        "For emergencies, call 911 immediately.**\n\n"
        "*⚡ Demo Mode active.*"
    ),
}


class MedGemmaEngineStandalone:
    def __init__(self):
        self._model = None
        self._processor = None
        self._current_model_id: Optional[str] = None
        self._demo_mode: bool = DEMO_MODE
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_demo(self):
        return self._demo_mode

    @property
    def loaded_model(self):
        return self._current_model_id

    def load_model(self, model_id: str) -> bool:
        if self._demo_mode:
            return False
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if self._device == "cuda" else None,
                token=HF_TOKEN or None,
                low_cpu_mem_usage=True,
            )
            self._processor = AutoProcessor.from_pretrained(
                model_id, token=HF_TOKEN or None
            )
            self._current_model_id = model_id
            self._demo_mode = False
            return True
        except Exception as e:
            self._demo_mode = True
            return False

    def _generate(self, messages: list, image_mode: bool = False) -> tuple[str, float]:
        t0 = time.perf_counter()
        if self._demo_mode or self._model is None:
            return "", (time.perf_counter() - t0) * 1000

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
            gen = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            gen = gen[0][input_len:]

        decoded = self._processor.decode(gen, skip_special_tokens=True)
        return decoded, (time.perf_counter() - t0) * 1000

    def analyze_image(self, image: Image.Image, prompt: str, img_type: str = "xray"):
        t0 = time.perf_counter()
        if self._demo_mode or self._model is None:
            return DEMO_RESPONSES.get(img_type, DEMO_RESPONSES["xray"]), True, 0.0
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text, ms = self._generate(messages, image_mode=True)
        if not text:
            return DEMO_RESPONSES.get(img_type, ""), True, ms
        return text, False, ms

    def medical_qa(self, question: str):
        t0 = time.perf_counter()
        if self._demo_mode or self._model is None:
            return DEMO_RESPONSES["medical_qa"], True, 0.0
        messages = [{"role": "user", "content": question}]
        text, ms = self._generate(messages)
        if not text:
            return DEMO_RESPONSES["medical_qa"], True, ms
        return text, False, ms

    def summarize_note(self, note: str):
        prompt = (
            "Analyze this clinical note and produce a structured summary with sections: "
            "Chief Complaint, History, Physical Exam, Assessment, Plan, Follow-up.\n\n"
            f"NOTE:\n{note}\n\nSUMMARY:"
        )
        t0 = time.perf_counter()
        if self._demo_mode or self._model is None:
            return DEMO_RESPONSES["clinical_note"], True, 0.0
        messages = [{"role": "user", "content": prompt}]
        text, ms = self._generate(messages)
        return text or DEMO_RESPONSES["clinical_note"], not bool(text), ms

    def triage(self, symptoms: str, patient_info: str):
        prompt = (
            "As a clinical decision support tool, assess triage acuity (Immediate/Emergent/"
            "Urgent/Semi-urgent/Non-urgent), red flags, and recommended action.\n\n"
            f"PATIENT: {patient_info}\nSYMPTOMS: {symptoms}\n\nASSESSMENT:"
        )
        if self._demo_mode or self._model is None:
            return DEMO_RESPONSES["patient_triage"], True, 0.0
        messages = [{"role": "user", "content": prompt}]
        text, ms = self._generate(messages)
        return text or DEMO_RESPONSES["patient_triage"], not bool(text), ms


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="MedGemma Clinical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a73e8,#0d47a1);padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;color:white}
.main-header h1{margin:0;font-size:2rem}
.main-header p{margin:.3rem 0 0;opacity:.85}
.result-box{background:#f8f9ff;border-left:4px solid #1a73e8;border-radius:0 8px 8px 0;padding:1.2rem 1.5rem;margin-top:1rem;font-size:.95rem;line-height:1.7}
.disclaimer{background:#fff3e0;border:1px solid #ff9800;border-radius:8px;padding:.8rem 1rem;font-size:.85rem;color:#e65100;margin-bottom:1rem}
.demo-badge{display:inline-block;background:#ff5722;color:white;padding:2px 10px;border-radius:12px;font-size:.75rem;font-weight:600}
.live-badge{display:inline-block;background:#4caf50;color:white;padding:2px 10px;border-radius:12px;font-size:.75rem;font-weight:600}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# Initialize engine in session state
if "engine" not in st.session_state:
    st.session_state.engine = MedGemmaEngineStandalone()
if "analyses" not in st.session_state:
    st.session_state.analyses = 0

engine: MedGemmaEngineStandalone = st.session_state.engine


# ---- Sidebar ----
with st.sidebar:
    st.markdown("<h2 style='color:white;text-align:center'>🏥 MedGemma AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#90caf9;text-align:center;font-size:.8rem'>Clinical AI Platform</p>", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📊 Dashboard", "🩻 Medical Imaging", "💬 Medical Q&A",
        "📋 Clinical Notes", "🚑 Patient Triage", "ℹ️ About"
    ], label_visibility="collapsed")

    st.divider()

    # Status
    mode_label = "⚡ Demo Mode" if engine.is_demo else "✅ Live Model"
    model_name = engine.loaded_model or "Not loaded"
    st.markdown(f"""
    <div style="background:#1565c0;border-radius:8px;padding:.8rem">
        <b style="color:white">System Status</b><br>
        <span style="color:#ffcc02">{mode_label}</span><br>
        <small style="color:#90caf9">{model_name}</small>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<b style='color:white;font-size:.9rem'>Load Model</b>", unsafe_allow_html=True)
    model_choice = st.selectbox("Model", [
        "google/medgemma-4b-it",
        "google/medgemma-1.5-4b-it",
        "google/medgemma-27b-text-it",
    ], label_visibility="collapsed")

    hf_tok = st.text_input("HuggingFace Token", type="password", placeholder="hf_...")

    if st.button("Load Model", use_container_width=True):
        if hf_tok:
            os.environ["HF_TOKEN"] = hf_tok
        with st.spinner("Loading — this may take several minutes..."):
            ok = engine.load_model(model_choice)
        if ok:
            st.success(f"Model loaded: {model_choice}")
            st.rerun()
        else:
            st.error("Load failed. Running in demo mode.")

    st.divider()
    st.markdown(
        "<p style='color:#90caf9;font-size:.72rem;text-align:center'>"
        "Powered by Google MedGemma<br>HAI-DEF Collection<br>"
        "MedGemma Impact Challenge</p>",
        unsafe_allow_html=True,
    )


# ---- Helper ----
def show_result(text: str, is_demo: bool, ms: float = 0.0):
    badge = '<span class="demo-badge">DEMO</span>' if is_demo else '<span class="live-badge">LIVE</span>'
    ms_str = f"<small>({ms:.0f} ms)</small>" if ms > 0 else ""
    st.markdown(f"**Result** {badge} {ms_str}", unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-box">{text.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )


# ==============================================================
# Pages
# ==============================================================

if "Dashboard" in page:
    st.markdown("""
    <div class="main-header">
        <h1>🏥 MedGemma Clinical AI</h1>
        <p>Human-centered AI for healthcare — MedGemma Impact Challenge Submission</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (str(st.session_state.analyses), "Analyses Run"),
        ("Demo" if engine.is_demo else "Live", "Mode"),
        ("4B / 27B", "Models"),
        ("4", "Modules"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.metric(label, val)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Platform Overview")
        st.markdown("""
This platform demonstrates real-world clinical AI capabilities powered by **Google MedGemma**
and the **Health AI Developer Foundations (HAI-DEF)** collection.

**Modules:**
- **Medical Imaging** — X-ray, dermatology, histopathology AI interpretation
- **Medical Q&A** — Evidence-based clinical Q&A (MedGemma 27B: 87.7% MedQA)
- **Clinical Notes** — Structured extraction from unstructured clinical text
- **Patient Triage** — AI-assisted symptom severity classification

**Privacy:** All inference runs locally. No patient data leaves your server.
        """)
    with col2:
        st.subheader("Quick Setup")
        st.code("""# Install dependencies
pip install -r requirements.txt

# Run standalone (demo mode)
streamlit run standalone_app.py

# With real model (requires GPU)
export HF_TOKEN=hf_your_token
export DEMO_MODE=false
streamlit run standalone_app.py

# Docker deployment
docker-compose up --build""", language="bash")

    st.divider()
    st.subheader("Model Performance")
    import pandas as pd
    df = pd.DataFrame({
        "Model": ["MedGemma 4B", "MedGemma 1.5 4B", "MedGemma 27B Text"],
        "MedQA Score": ["64.4%", "~67%", "87.7%"],
        "Modality": ["Multimodal", "Multimodal (v1.5)", "Text only"],
        "GPU RAM": ["~8 GB", "~8 GB", "~55 GB"],
        "Use Case": ["Imaging + QA", "Advanced Imaging", "Medical Q&A"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


elif "Medical Imaging" in page:
    st.title("🩻 Medical Image Analysis")
    st.markdown('<div class="disclaimer">⚠️ For research only. Not a substitute for professional diagnosis.</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🫁 Chest X-Ray", "🔬 Dermatology", "🧫 Histopathology"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            up = st.file_uploader("Upload X-Ray (JPEG/PNG)", type=["jpg","jpeg","png"], key="xr")
            ctx = st.text_area("Clinical context", placeholder="Age, symptoms, history...", height=70)
            if up: st.image(Image.open(up), use_container_width=True)
            btn = st.button("Analyze X-Ray", type="primary", use_container_width=True, disabled=not up)
        with c2:
            if btn and up:
                img = Image.open(up).convert("RGB")
                prompt = (
                    "You are an expert radiologist. Provide a structured chest X-ray report including: "
                    "1) Systematic review (lungs, heart, mediastinum, bones) "
                    "2) Abnormalities identified "
                    "3) Impression and recommendation."
                    + (f" Clinical context: {ctx}" if ctx else "")
                )
                with st.spinner("Analyzing with MedGemma..."):
                    text, demo, ms = engine.analyze_image(img, prompt, "xray")
                st.session_state.analyses += 1
                show_result(text, demo, ms)
                st.download_button("Download Report", text, "xray_report.txt")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            up = st.file_uploader("Upload Skin Lesion", type=["jpg","jpeg","png"], key="dm")
            hist = st.text_area("Lesion history", placeholder="Duration, changes...", height=70)
            if up: st.image(Image.open(up), use_container_width=True)
            btn = st.button("Analyze Lesion", type="primary", use_container_width=True, disabled=not up)
        with c2:
            if btn and up:
                img = Image.open(up).convert("RGB")
                prompt = (
                    "You are a board-certified dermatologist. Analyze this skin lesion using ABCDE criteria, "
                    "provide morphological description, top 3 differential diagnoses, and management recommendation."
                    + (f" History: {hist}" if hist else "")
                )
                with st.spinner("Analyzing..."):
                    text, demo, ms = engine.analyze_image(img, prompt, "dermatology")
                st.session_state.analyses += 1
                show_result(text, demo, ms)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            up = st.file_uploader("Upload Slide Image", type=["jpg","jpeg","png"], key="ph")
            stain = st.selectbox("Stain", ["H&E","IHC","PAS","Trichrome","Other"])
            cinfo = st.text_area("Clinical info", height=70)
            if up: st.image(Image.open(up), use_container_width=True)
            btn = st.button("Analyze Slide", type="primary", use_container_width=True, disabled=not up)
        with c2:
            if btn and up:
                img = Image.open(up).convert("RGB")
                prompt = (
                    f"You are an expert pathologist. Analyze this {stain}-stained histopathology slide. "
                    "Describe tissue architecture, cellular morphology, mitotic count, grade if applicable, "
                    "and provide diagnosis with differential."
                    + (f" Clinical context: {cinfo}" if cinfo else "")
                )
                with st.spinner("Analyzing..."):
                    text, demo, ms = engine.analyze_image(img, prompt, "pathology")
                st.session_state.analyses += 1
                show_result(text, demo, ms)


elif "Medical Q&A" in page:
    st.title("💬 Medical Q&A")
    st.markdown('<div class="disclaimer">⚠️ Educational purposes only. Not clinical advice.</div>',
                unsafe_allow_html=True)

    SAMPLES = [
        "What are the diagnostic criteria for Type 2 Diabetes (ADA guidelines)?",
        "Explain the mechanism of action of beta-blockers in heart failure.",
        "What are the first-line treatments for community-acquired pneumonia?",
        "Describe the pathophysiology of acute kidney injury.",
        "What are the indications for prophylactic anticoagulation in hospitalized patients?",
    ]

    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_area("Clinical Question", height=130,
                         placeholder="Enter your medical question...")
        ctx2 = st.text_area("Context (optional)", height=70,
                            placeholder="Patient demographics, relevant history...")
        if st.button("Get Answer", type="primary", use_container_width=True, disabled=not q.strip()):
            full_q = f"Context: {ctx2}\n\nQuestion: {q}" if ctx2.strip() else q
            with st.spinner("MedGemma is analyzing..."):
                text, demo, ms = engine.medical_qa(full_q)
            st.session_state.analyses += 1
            show_result(text, demo, ms)
            st.download_button("Download Response", f"Q: {q}\n\nA: {text}", "qa_response.txt")

    with c2:
        st.markdown("**Sample Questions**")
        for i, sq in enumerate(SAMPLES):
            st.button(sq[:55] + "…", key=f"sq{i}", disabled=True)

        st.divider()
        st.markdown("""
**Benchmark Results:**
| Benchmark | 27B Score |
|-----------|-----------|
| MedQA | 87.7% |
| MedMCQA | 73.4% |
| PubMedQA | 79.3% |
        """)


elif "Clinical Notes" in page:
    st.title("📋 Clinical Note Summarizer")
    st.markdown('<div class="disclaimer">⚠️ De-identify patient data before use. Research only.</div>',
                unsafe_allow_html=True)

    SAMPLE = """Date: 2024-01-15 | Dr. Smith

Chief Complaint: 68M with 3-day progressive dyspnea and productive cough.

HPI: Worsening dyspnea on exertion, yellow sputum, fever to 38.8°C. No chest pain.
PMH: COPD (moderate, FEV1 62%), HTN, T2DM (HbA1c 7.2%).
Meds: Tiotropium, Albuterol PRN, Lisinopril 10mg, Metformin 1000mg BID.

Vitals: BP 142/88, HR 98, RR 22, SpO2 91% RA, Temp 38.6°C.
Exam: Mild respiratory distress. Decreased air entry right base. Dullness right lower lobe.
Labs: WBC 14.2, CRP 85, PCT 0.8.

Assessment: Community-acquired pneumonia (CAP) right lower lobe, moderate (PSI Class III). COPD exacerbation.

Plan:
1. IV Ceftriaxone 1g Q24h + Azithromycin 500mg QD x5d
2. O2 to maintain SpO2 >94%
3. Albuterol nebs Q4h
4. Prednisone 40mg QD x5d
5. CXR baseline, repeat in 4-6 weeks
6. Monitor glucose (steroids)
7. Follow-up 1 week PCP"""

    c1, c2 = st.columns(2)
    with c1:
        use_sample = st.checkbox("Use sample note")
        note = st.text_area("Clinical Note (de-identified)", value=SAMPLE if use_sample else "",
                            height=380)
        if st.button("Summarize & Extract", type="primary", use_container_width=True,
                     disabled=not note.strip()):
            with st.spinner("Processing with MedGemma NLP..."):
                text, demo, ms = engine.summarize_note(note)
            st.session_state.analyses += 1
            show_result(text, demo, ms)
            st.download_button("Download Summary", text, "clinical_summary.txt")
    with c2:
        st.info(
            "Extracts structured sections:\n"
            "- Chief Complaint\n"
            "- History of Present Illness\n"
            "- Physical Examination\n"
            "- Assessment & Diagnoses\n"
            "- Treatment Plan\n"
            "- Follow-up Items"
        )


elif "Patient Triage" in page:
    st.title("🚑 Patient Triage Assistant")
    st.error("⚠️ DEMO ONLY — NOT for real triage decisions. Call 911 for emergencies.")

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 0, 120, 45)
        sex = st.selectbox("Sex", ["Male","Female","Other"])
        symptoms = st.text_area("Describe symptoms", height=150,
                                placeholder="Be specific: onset, location, severity, associated symptoms...")
        cv1, cv2 = st.columns(2)
        with cv1:
            bp = st.text_input("BP (mmHg)", "120/80")
            hr = st.number_input("HR (bpm)", 0, 300, 80)
        with cv2:
            spo2 = st.number_input("SpO2 (%)", 0, 100, 98)
            temp = st.number_input("Temp (°C)", 30.0, 45.0, 37.0)
        comorbid = st.multiselect("Comorbidities",
                                  ["Diabetes","HTN","Heart disease","COPD","CKD","Immunocompromised","None"],
                                  default=["None"])

        triage_btn = st.button("Assess Triage", type="primary", use_container_width=True,
                               disabled=not symptoms.strip())
    with c2:
        if triage_btn and symptoms.strip():
            patient_info = (
                f"{age}yo {sex}. Comorbidities: {', '.join(comorbid)}. "
                f"Vitals: BP {bp}, HR {hr}, SpO2 {spo2}%, Temp {temp}°C."
            )
            with st.spinner("Performing triage assessment..."):
                text, demo, ms = engine.triage(symptoms, patient_info)
            st.session_state.analyses += 1
            show_result(text, demo, ms)
            st.error("Reminder: Always seek professional medical care for real emergencies.")
        else:
            st.markdown("""
**Emergency Triage Scale:**

🔴 **Immediate** — Life-threatening, resuscitation needed
🟠 **Emergent** — Seen within 15 minutes
🟡 **Urgent** — Seen within 30 minutes
🟢 **Semi-urgent** — Seen within 1 hour
⚪ **Non-urgent** — Seen within 2 hours

*Fill in symptoms and click "Assess Triage".*
            """)


elif "About" in page:
    st.title("ℹ️ About MedGemma Clinical AI")
    st.markdown("""
## MedGemma Impact Challenge Submission

Built for the **[MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)**
hosted by Google Research on Kaggle ($100,000 prize pool, deadline Feb 24 2026).

---

## Models (HAI-DEF Collection)

| Model | Size | MedQA | Key Capability |
|-------|------|-------|----------------|
| `medgemma-4b-it` | 4B | 64.4% | Multimodal image+text |
| `medgemma-1.5-4b-it` | 4B | ~67% | Advanced imaging (CT/MRI) |
| `medgemma-27b-text-it` | 27B | **87.7%** | Medical Q&A, text |

---

## Architecture

```
Streamlit Frontend (port 8501)
    │
    ├── Medical Imaging (X-ray, Dermatology, Pathology)
    ├── Medical Q&A (MedGemma 27B)
    ├── Clinical Note NLP
    └── Patient Triage
         │
         ▼
FastAPI Backend (port 8000)
         │
         ▼
MedGemma Engine (HuggingFace Transformers)
    ├── google/medgemma-4b-it (multimodal)
    └── google/medgemma-27b-text-it (text)
```

---

## References
- [MedGemma Google Developers](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [HAI-DEF HuggingFace Collection](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def)
- [MedGemma Research Blog](https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/)
- [MedGemma GitHub](https://github.com/Google-Health/medgemma)
    """)
