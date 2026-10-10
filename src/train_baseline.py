"""
Baseline Model Training Pipeline for Mailora Email Classifier (v1.1.0).

Trains an optimized scikit-learn pipeline using TF-IDF vectorization
(unigrams, bigrams, and trigrams + character n-grams) and Logistic Regression.
Tuned to avoid false positives on educational newsletters and industry discussions
while maintaining high recall on actionable career opportunities.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.preprocess import combine_subject_and_body
from src.data_preparation import prepare_and_split_dataset, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_VERSION = "1.1.0"


def build_feature_extractor() -> FeatureUnion:
    """
    Constructs a dual-feature TF-IDF union combining:
    1. Word-level n-grams (1 to 3) capturing actionable intent phrases
       (e.g., 'apply now at', 'interview scheduled for', 'online assessment invitation')
       versus informational markers ('why ai hasn t', 'how to crack', 'salary report for').
    2. Character n-grams (3 to 5) within word boundaries for acronyms (SDE, CTC, PNR, GET, OTP).
    """
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=12000,
        sublinear_tf=True,
        stop_words="english",
        min_df=2
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=6000,
        sublinear_tf=True,
        min_df=2
    )

    union = FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
        ],
        n_jobs=-1
    )
    return union


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combines subject and body into fused input text, and extracts labels.
    """
    texts = [
        combine_subject_and_body(row["subject"], row["body"], subject_weight=2)
        for _, row in df.iterrows()
    ]
    labels = df["label"].values.astype(int)
    return np.array(texts), labels


def find_optimal_threshold(
    pipeline: Pipeline,
    val_texts: np.ndarray,
    val_labels: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """
    Calibrates the probability decision threshold on the VALIDATION set.
    Balances precision and recall to eliminate false positives on career-adjacent
    content (newsletters, blogs, salary surveys) while preserving high opportunity recall.
    """
    val_probs = pipeline.predict_proba(val_texts)[:, 1]

    best_thresh = 0.50
    best_f1 = -1.0
    best_metrics = {}

    # Sweep thresholds from 0.35 to 0.70
    for thresh in np.arange(0.35, 0.71, 0.02):
        preds = (val_probs >= thresh).astype(int)
        rec = recall_score(val_labels, preds, zero_division=0)
        prec = precision_score(val_labels, preds, zero_division=0)
        f1 = f1_score(val_labels, preds, zero_division=0)

        # We want balanced high F1 with recall >= 0.92
        if rec >= 0.92 and prec >= 0.92:
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
                best_metrics = {
                    "threshold": round(best_thresh, 3),
                    "recall": round(rec, 4),
                    "precision": round(prec, 4),
                    "f1": round(f1, 4),
                    "accuracy": round(accuracy_score(val_labels, preds), 4)
                }

    # Fallback to pure F1 maximization if constrained window didn't trigger
    if not best_metrics:
        for thresh in np.arange(0.35, 0.71, 0.05):
            preds = (val_probs >= thresh).astype(int)
            f1 = f1_score(val_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
                best_metrics = {
                    "threshold": round(best_thresh, 3),
                    "recall": round(recall_score(val_labels, preds, zero_division=0), 4),
                    "precision": round(precision_score(val_labels, preds, zero_division=0), 4),
                    "f1": round(f1, 4),
                    "accuracy": round(accuracy_score(val_labels, preds), 4)
                }

    logger.info(f"Calibrated Threshold on Validation Set: {best_thresh:.3f} -> {best_metrics}")
    return best_thresh, best_metrics


def train_baseline_model(
    dataset_dir: str = "dataset",
    artifacts_dir: str = "artifacts",
    random_seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Complete end-to-end training and serialization pipeline for baseline model.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load splits
    train_csv = os.path.join(dataset_dir, "train.csv")
    val_csv = os.path.join(dataset_dir, "val.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")

    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        logger.info("Splits not found on disk. Preparing fresh dataset...")
        train_df, val_df, test_df = prepare_and_split_dataset(output_dir=dataset_dir)
    else:
        logger.info(f"Loading datasets from {dataset_dir}")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)

    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # 2. Build Pipeline
    feature_union = build_feature_extractor()
    classifier = LogisticRegression(
        C=1.2,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_seed,
        max_iter=1000
    )

    pipeline = Pipeline([
        ("features", feature_union),
        ("classifier", classifier)
    ])

    # 3. Stratified 5-Fold Cross-Validation on Training Data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    logger.info(f"5-Fold CV F1 Scores: {cv_scores.round(4)} (Mean: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f})")

    # 4. Fit Pipeline on full training data
    logger.info("Fitting updated baseline model pipeline...")
    pipeline.fit(X_train, y_train)

    # 5. Calibrate threshold on validation data (NEVER on test set)
    calibrated_threshold, val_metrics = find_optimal_threshold(pipeline, X_val, y_val)

    # 6. Evaluate on Unseen Independent Test Set
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= calibrated_threshold).astype(int)

    test_accuracy = float(accuracy_score(y_test, test_preds))
    test_precision = float(precision_score(y_test, test_preds, zero_division=0))
    test_recall = float(recall_score(y_test, test_preds, zero_division=0))
    test_f1 = float(f1_score(y_test, test_preds, zero_division=0))
    test_roc_auc = float(roc_auc_score(y_test, test_probs))
    cm = confusion_matrix(y_test, test_preds).tolist()
    report = classification_report(
        y_test,
        test_preds,
        target_names=["NON_CAREER_RELATED", "CAREER_RELATED"],
        output_dict=True
    )

    # Calculate False Positive Rate and False Negatives
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    logger.info("=== Independent Test Set Evaluation (v1.1.0) ===")
    logger.info(f"Accuracy:  {test_accuracy:.4f}")
    logger.info(f"Precision: {test_precision:.4f}")
    logger.info(f"Recall:    {test_recall:.4f}")
    logger.info(f"F1-Score:  {test_f1:.4f}")
    logger.info(f"ROC-AUC:   {test_roc_auc:.4f}")
    logger.info(f"False Positive Rate: {fpr:.4f} (FP: {fp}, TN: {tn})")
    logger.info(f"False Negatives:     {fn} (TP: {tp})")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, test_preds)}")

    # 7. Serialize Artifacts
    model_artifact_path = os.path.join(artifacts_dir, "baseline_pipeline.joblib")
    joblib.dump(pipeline, model_artifact_path)
    logger.info(f"Saved serialized model pipeline to {model_artifact_path}")

    model_size_mb = os.path.getsize(model_artifact_path) / (1024 * 1024)

    # Metadata & Evaluation Results
    metadata = {
        "model_type": "baseline_tfidf_logistic_regression",
        "model_version": MODEL_VERSION,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "random_seed": random_seed,
        "calibrated_threshold": calibrated_threshold,
        "model_artifact": os.path.basename(model_artifact_path),
        "model_size_mb": round(model_size_mb, 2),
        "cross_validation_f1_mean": round(float(cv_scores.mean()), 4),
        "cross_validation_f1_std": round(float(cv_scores.std()), 4),
        "validation_metrics": val_metrics,
        "test_metrics": {
            "accuracy": round(test_accuracy, 4),
            "precision": round(test_precision, 4),
            "recall": round(test_recall, 4),
            "f1_score": round(test_f1, 4),
            "roc_auc": round(test_roc_auc, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negatives": fn,
            "false_positives": fp,
            "true_positives": tp,
            "true_negatives": tn,
            "confusion_matrix": cm,
            "classification_report": report
        }
    }

    metadata_path = os.path.join(artifacts_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata and evaluation to {metadata_path}")

    return metadata


if __name__ == "__main__":
    train_baseline_model()
