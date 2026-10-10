"""
Inference Engine for Mailora Email Classifier.

Supports both Baseline (TF-IDF + Logistic Regression) and Transformer (DistilBERT)
backends with dynamic model loading, calibrated thresholding, and genuine confidence scores.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
import joblib
import numpy as np

from src.preprocess import clean_text, combine_subject_and_body

logger = logging.getLogger(__name__)


class EmailPredictor:
    """
    Unified predictor supporting both baseline and transformer classifiers.
    """
    def __init__(
        self,
        model_type: str = "baseline",
        artifacts_dir: str = "artifacts",
        threshold_override: Optional[float] = None
    ):
        self.model_type = model_type
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.threshold = 0.50
        self.model_version = "1.0.0"
        self.is_loaded = False

        self._load_model(threshold_override)

    def _load_model(self, threshold_override: Optional[float] = None):
        """Loads model weights, tokenizer, and calibration metadata."""
        if self.model_type == "baseline":
            model_path = os.path.join(self.artifacts_dir, "baseline_pipeline.joblib")
            meta_path = os.path.join(self.artifacts_dir, "model_metadata.json")

            if not os.path.exists(model_path):
                logger.warning(f"Baseline model not found at {model_path}. Predictor in uninitialized state.")
                return

            self.model = joblib.load(model_path)
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.threshold = meta.get("calibrated_threshold", 0.30)
                    self.model_version = meta.get("model_version", "1.0.0")

            if threshold_override is not None:
                self.threshold = threshold_override

            self.is_loaded = True
            logger.info(f"Loaded Baseline Model v{self.model_version} (Threshold: {self.threshold})")

        elif self.model_type == "transformer":
            transformer_dir = os.path.join(self.artifacts_dir, "distilbert")
            meta_path = os.path.join(transformer_dir, "transformer_metadata.json")

            if not os.path.exists(transformer_dir):
                logger.warning(f"DistilBERT model not found at {transformer_dir}. Falling back to baseline.")
                self.model_type = "baseline"
                self._load_model(threshold_override)
                return

            import torch
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(transformer_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(transformer_dir)
            self.model.to(self.device)
            self.model.eval()

            self.threshold = 0.50
            if threshold_override is not None:
                self.threshold = threshold_override

            self.model_version = "1.0.0-distilbert"
            self.is_loaded = True
            logger.info(f"Loaded DistilBERT Model on {self.device} (Threshold: {self.threshold})")

    def predict_single(self, subject: Optional[str], body: Optional[str]) -> Dict[str, Any]:
        """
        Classifies a single email into CAREER_RELATED or NON_CAREER_RELATED.
        Returns label, career boolean, genuine confidence score, and latency.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Please train or verify artifacts directory.")

        start_time = time.perf_counter()

        # Input sanitization and preprocessing
        safe_subject = str(subject or "")
        safe_body = str(body or "")
        fused_text = combine_subject_and_body(safe_subject, safe_body, subject_weight=2)

        if not fused_text.strip():
            # Handle empty content safely
            return {
                "label": "NON_CAREER_RELATED",
                "is_career_related": False,
                "confidence": 0.50,
                "career_probability": 0.0,
                "model_version": self.model_version,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

        career_prob = 0.0

        if self.model_type == "baseline":
            # Scikit-learn Pipeline
            probs = self.model.predict_proba([fused_text])[0]
            career_prob = float(probs[1])

        elif self.model_type == "transformer":
            import torch
            inputs = self.tokenizer(
                [fused_text],
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
                career_prob = float(probs[1])

        # Classification decision based on calibrated threshold
        is_career = bool(career_prob >= self.threshold)
        label = "CAREER_RELATED" if is_career else "NON_CAREER_RELATED"

        # Confidence: continuous, un-hardcoded posterior probability for the chosen class
        # If predicted career: confidence is career_prob.
        # If predicted non-career: confidence is (1.0 - career_prob).
        confidence = career_prob if is_career else (1.0 - career_prob)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "label": label,
            "is_career_related": is_career,
            "confidence": round(confidence, 4),
            "career_probability": round(career_prob, 4),
            "model_version": self.model_version,
            "latency_ms": round(elapsed_ms, 2)
        }

    def predict_batch(self, emails: List[Dict[str, Optional[str]]]) -> List[Dict[str, Any]]:
        """
        Classifies a batch of emails efficiently.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        if not emails:
            return []

        start_time = time.perf_counter()

        fused_texts = [
            combine_subject_and_body(str(e.get("subject") or ""), str(e.get("body") or ""), subject_weight=2)
            for e in emails
        ]

        if self.model_type == "baseline":
            probs = self.model.predict_proba(fused_texts)[:, 1]
        elif self.model_type == "transformer":
            import torch
            inputs = self.tokenizer(
                fused_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

        results = []
        total_time_ms = (time.perf_counter() - start_time) * 1000
        per_item_ms = round(total_time_ms / len(emails), 2)

        for prob in probs:
            career_prob = float(prob)
            is_career = bool(career_prob >= self.threshold)
            label = "CAREER_RELATED" if is_career else "NON_CAREER_RELATED"
            confidence = career_prob if is_career else (1.0 - career_prob)

            results.append({
                "label": label,
                "is_career_related": is_career,
                "confidence": round(confidence, 4),
                "career_probability": round(career_prob, 4),
                "model_version": self.model_version,
                "latency_ms": per_item_ms
            })

        return results
