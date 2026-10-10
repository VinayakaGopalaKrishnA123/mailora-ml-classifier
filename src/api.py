"""
Production FastAPI Classification Service for Mailora.

Exposes REST endpoints:
- GET  /health          : Service health & loaded model status
- POST /predict         : Single email classification with confidence
- POST /predict/batch   : High-throughput batch classification
- POST /feedback        : Consent-based feedback loop for model improvement
"""

import os
import time
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.predict import EmailPredictor

# Configure privacy-preserving logger (never logs raw email bodies)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MailoraClassifierAPI")

# Global predictor and service state
predictor: Optional[EmailPredictor] = None
service_start_time: float = time.time()

# Environment variables
MODEL_TYPE = os.getenv("MODEL_TYPE", "baseline")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
FEEDBACK_STORE_PATH = os.getenv("FEEDBACK_STORE_PATH", "artifacts/user_feedback.jsonl")
ENABLE_FEEDBACK_STORAGE = os.getenv("ENABLE_FEEDBACK_STORAGE", "true").lower() == "true"
MAX_SUBJECT_LENGTH = int(os.getenv("MAX_SUBJECT_LENGTH", "500"))
MAX_BODY_LENGTH = int(os.getenv("MAX_BODY_LENGTH", "50000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the ML model predictor at startup."""
    global predictor
    logger.info(f"Initializing EmailPredictor (Type: {MODEL_TYPE}, Artifacts: {ARTIFACTS_DIR})...")
    try:
        predictor = EmailPredictor(model_type=MODEL_TYPE, artifacts_dir=ARTIFACTS_DIR)
        logger.info(f"Predictor successfully initialized (Loaded: {predictor.is_loaded}).")
    except Exception as exc:
        logger.error(f"Failed to initialize predictor: {exc}", exc_info=True)
        predictor = None
    yield
    logger.info("Shutting down Mailora Classifier API.")


app = FastAPI(
    title="Mailora Email Classification API",
    description=(
        "Production-ready NLP service for classifying emails into CAREER_RELATED "
        "and NON_CAREER_RELATED categories for the Mailora Career Opportunity Tracking System."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request and Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class EmailPredictRequest(BaseModel):
    subject: Optional[str] = Field(
        default="",
        max_length=MAX_SUBJECT_LENGTH,
        description="The subject line of the email",
        example="Software Engineer Internship - Applications Open"
    )
    body: Optional[str] = Field(
        default="",
        max_length=MAX_BODY_LENGTH,
        description="The body text or snippet of the email",
        example="We are inviting applications for our summer software engineering internship program..."
    )

    @field_validator("subject", "body", mode="before")
    def sanitize_strings(cls, v):
        if v is None:
            return ""
        return str(v)


class EmailPredictResponse(BaseModel):
    label: str = Field(description="'CAREER_RELATED' or 'NON_CAREER_RELATED'")
    is_career_related: bool = Field(description="True if email is career-related, False otherwise")
    confidence: float = Field(
        description=(
            "Calculated posterior confidence score (0.50 to 1.00) indicating the model's "
            "certainty in the predicted category. Derived from calibrated probability distribution."
        )
    )
    career_probability: float = Field(description="Direct raw probability of belonging to CAREER_RELATED class")
    model_version: str = Field(description="Version of the model producing this classification")
    latency_ms: float = Field(description="Inference latency in milliseconds")


class BatchPredictRequest(BaseModel):
    emails: List[EmailPredictRequest] = Field(
        min_length=1,
        max_length=100,
        description="List of email objects to classify (max 100 per batch)"
    )


class BatchPredictResponse(BaseModel):
    predictions: List[EmailPredictResponse]
    total_emails: int
    total_latency_ms: float


class FeedbackRequest(BaseModel):
    email_id: Optional[str] = Field(
        default=None,
        description="Safe identifier, anonymized token, or hash of the email"
    )
    subject: Optional[str] = Field(default="", max_length=MAX_SUBJECT_LENGTH)
    predicted_label: str = Field(description="The predicted label ('CAREER_RELATED' or 'NON_CAREER_RELATED')")
    corrected_label: str = Field(description="User-corrected label ('CAREER_RELATED' or 'NON_CAREER_RELATED')")
    user_consent: bool = Field(
        default=True,
        description="Explicit user consent to store feedback for dataset and model improvement"
    )
    feedback_notes: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("predicted_label", "corrected_label")
    def validate_labels(cls, v):
        v = v.strip().upper()
        if v not in {"CAREER_RELATED", "NON_CAREER_RELATED"}:
            raise ValueError("Label must be either 'CAREER_RELATED' or 'NON_CAREER_RELATED'")
        return v


class FeedbackResponse(BaseModel):
    status: str
    message: str
    recorded_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    model_version: str
    threshold: float
    uptime_seconds: float


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Returns service health and model initialization status.
    """
    global predictor
    is_loaded = predictor is not None and predictor.is_loaded
    uptime = round(time.time() - service_start_time, 1)

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_type=predictor.model_type if is_loaded else "none",
        model_version=predictor.model_version if is_loaded else "unknown",
        threshold=predictor.threshold if is_loaded else 0.50,
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=EmailPredictResponse, tags=["Classification"])
async def predict_email(request: EmailPredictRequest):
    """
    Classifies a single incoming email into CAREER_RELATED or NON_CAREER_RELATED.
    """
    global predictor
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification model is not loaded or currently unavailable."
        )

    # Validate that at least some content is provided
    if not request.subject.strip() and not request.body.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both 'subject' and 'body' are empty. Provide at least one field."
        )

    try:
        result = predictor.predict_single(request.subject, request.body)
        logger.info(
            f"Prediction completed: label={result['label']}, "
            f"confidence={result['confidence']}, latency={result['latency_ms']}ms"
        )
        return EmailPredictResponse(**result)
    except Exception as exc:
        logger.error(f"Inference error during /predict: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process classification request."
        )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Classification"])
async def predict_batch(request: BatchPredictRequest):
    """
    Classifies multiple emails efficiently in a single batched request.
    """
    global predictor
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification model is not loaded."
        )

    start_batch = time.perf_counter()
    email_items = [{"subject": e.subject, "body": e.body} for e in request.emails]

    try:
        predictions = predictor.predict_batch(email_items)
        elapsed_total_ms = round((time.perf_counter() - start_batch) * 1000, 2)

        logger.info(f"Batch prediction completed: {len(email_items)} emails in {elapsed_total_ms}ms")
        return BatchPredictResponse(
            predictions=[EmailPredictResponse(**p) for p in predictions],
            total_emails=len(predictions),
            total_latency_ms=elapsed_total_ms
        )
    except Exception as exc:
        logger.error(f"Batch inference error: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch classification execution failed."
        )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Active Learning"])
async def record_feedback(feedback: FeedbackRequest):
    """
    Accepts user feedback / corrections for misclassified emails.
    Stores safe metadata and anonymized features when user consent is provided.
    Privacy guarantee: Raw email bodies are NEVER stored in feedback logs.
    """
    if not feedback.user_consent:
        return FeedbackResponse(
            status="skipped",
            message="Feedback not recorded because user consent was not granted.",
            recorded_at=datetime.utcnow().isoformat() + "Z"
        )

    if not ENABLE_FEEDBACK_STORAGE:
        return FeedbackResponse(
            status="disabled",
            message="Feedback storage is disabled by server configuration.",
            recorded_at=datetime.utcnow().isoformat() + "Z"
        )

    # Anonymized email identifier or SHA256 signature
    safe_id = feedback.email_id or hashlib.sha256(
        (feedback.subject or "").encode("utf-8")
    ).hexdigest()[:16]

    feedback_record = {
        "email_ref": safe_id,
        "subject_snippet": feedback.subject[:80] if feedback.subject else "",
        "predicted_label": feedback.predicted_label,
        "corrected_label": feedback.corrected_label,
        "notes": feedback.feedback_notes,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    try:
        os.makedirs(os.path.dirname(FEEDBACK_STORE_PATH), exist_ok=True)
        with open(FEEDBACK_STORE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_record) + "\n")

        logger.info(f"Recorded correction for email [{safe_id}]: {feedback.predicted_label} -> {feedback.corrected_label}")
        return FeedbackResponse(
            status="recorded",
            message="Feedback stored successfully for model improvement.",
            recorded_at=feedback_record["timestamp"]
        )
    except Exception as exc:
        logger.error(f"Failed to record user feedback: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not persist feedback record."
        )
