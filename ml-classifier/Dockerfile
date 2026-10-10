# ==============================================================================
# Mailora Email Classifier - Production Dockerfile (Baseline v1.1.0)
# Lightweight, high-throughput microservice (~4.5ms inference)
# ==============================================================================

FROM python:3.10-slim AS builder

WORKDIR /app

# Install system build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Create wheel cache and install
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.3.0 \
    joblib>=1.3.0 \
    fastapi>=0.100.0 \
    "uvicorn[standard]>=0.23.0" \
    pydantic>=2.0.0 \
    python-dotenv>=1.0.0 \
    httpx>=0.24.0

# ------------------------------------------------------------------------------
# Production Runtime Stage
# ------------------------------------------------------------------------------
FROM python:3.10-slim AS runner

WORKDIR /app

# Create non-privileged user for security
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/sh appuser

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code and model artifacts
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/
COPY dataset/ /app/dataset/

# Set environment defaults
ENV PORT=8000 \
    HOST=0.0.0.0 \
    MODEL_TYPE=baseline \
    MODEL_DIR=/app/artifacts \
    WORKERS=2 \
    PYTHONUNBUFFERED=1

# Expose service port
EXPOSE 8000

# Grant permissions to non-root app user
RUN chown -R appuser:appgroup /app
USER appuser

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI uvicorn production server
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-2}"]
