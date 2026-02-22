# ===================================================================
# MedGemma Clinical AI — Multi-stage Dockerfile
# ===================================================================
# Stage 1: Base Python with CUDA support
FROM python:3.11-slim AS base

LABEL maintainer="MedGemma Impact Challenge Submission"
LABEL description="MedGemma Clinical AI Platform"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 medgemma
WORKDIR /app

# ===================================================================
# Stage 2: Install Python dependencies
# ===================================================================
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===================================================================
# Stage 3: Backend service
# ===================================================================
FROM deps AS backend

COPY backend/ ./backend/

USER medgemma
WORKDIR /app/backend

ENV PYTHONPATH=/app/backend
ENV PYTHONUNBUFFERED=1
ENV DEMO_MODE=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ===================================================================
# Stage 4: Frontend service
# ===================================================================
FROM deps AS frontend

COPY frontend/ ./frontend/
COPY backend/ ./backend/

USER medgemma
WORKDIR /app/frontend

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=http://backend:8000

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
