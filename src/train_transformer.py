"""
Stage B: DistilBERT Binary Text Classifier Training Pipeline.

Fine-tunes a pretrained DistilBERT model for binary email classification
(CAREER_RELATED vs NON_CAREER_RELATED) using PyTorch and Hugging Face Transformers.
Supports both CPU and GPU execution, handling long email truncation and
safe checkpoint saving.
"""

import os
import json
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
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
from src.data_preparation import RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Training hyperparameters
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


class EmailDataset(Dataset):
    """PyTorch Dataset for tokenized email classification inputs."""
    def __init__(self, texts: list, labels: list, tokenizer: DistilBertTokenizerFast, max_length: int = MAX_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def set_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_transformer_model(
    dataset_dir: str = "dataset",
    output_dir: str = "artifacts/distilbert",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_length: int = MAX_LENGTH
) -> Dict[str, Any]:
    """
    Fine-tunes DistilBERT on the training set and evaluates on the test set.
    """
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for training: {device}")

    # Memory and training documentation
    logger.info(f"Transformer Configuration:")
    logger.info(f"  Base Model: {MODEL_NAME}")
    logger.info(f"  Max Sequence Length: {max_length}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Hardware: {'CUDA GPU' if torch.cuda.is_available() else 'Standard CPU (Estimated memory: ~1.5 GB RAM)'}")

    # Load data
    train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(dataset_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(dataset_dir, "test.csv"))

    train_texts = [combine_subject_and_body(r["subject"], r["body"]) for _, r in train_df.iterrows()]
    val_texts = [combine_subject_and_body(r["subject"], r["body"]) for _, r in val_df.iterrows()]
    test_texts = [combine_subject_and_body(r["subject"], r["body"]) for _, r in test_df.iterrows()]

    train_labels = train_df["label"].tolist()
    val_labels = val_df["label"].tolist()
    test_labels = test_df["label"].tolist()

    # Load Tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = EmailDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = EmailDataset(val_texts, val_labels, tokenizer, max_length=max_length)
    test_dataset = EmailDataset(test_texts, test_labels, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    logger.info(f"Loading pretrained model: {MODEL_NAME}...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NON_CAREER_RELATED", 1: "CAREER_RELATED"},
        label2id={"NON_CAREER_RELATED": 0, "CAREER_RELATED": 1}
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # Training Loop
    logger.info("Beginning fine-tuning...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Step
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch["labels"].numpy())

        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")

    # Evaluate on Unseen Test Set
    model.eval()
    test_probs, test_preds, test_targets = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_probs.extend(probs)
            test_preds.extend(preds)
            test_targets.extend(batch["labels"].numpy())

    test_acc = float(accuracy_score(test_targets, test_preds))
    test_prec = float(precision_score(test_targets, test_preds, zero_division=0))
    test_rec = float(recall_score(test_targets, test_preds, zero_division=0))
    test_f1 = float(f1_score(test_targets, test_preds, zero_division=0))
    test_roc = float(roc_auc_score(test_targets, test_probs))
    cm = confusion_matrix(test_targets, test_preds).tolist()
    report = classification_report(
        test_targets,
        test_preds,
        target_names=["NON_CAREER_RELATED", "CAREER_RELATED"],
        output_dict=True
    )

    logger.info("=== Transformer Test Set Evaluation ===")
    logger.info(f"Accuracy:  {test_acc:.4f}")
    logger.info(f"Precision: {test_prec:.4f}")
    logger.info(f"Recall:    {test_rec:.4f}")
    logger.info(f"F1-Score:  {test_f1:.4f}")
    logger.info(f"ROC-AUC:   {test_roc:.4f}")

    # Save Model & Tokenizer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved DistilBERT model and tokenizer to {output_dir}")

    metadata = {
        "model_type": "distilbert_transformer",
        "model_version": "1.1.0-distilbert",
        "model_name": MODEL_NAME,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "test_metrics": {
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1_score": round(test_f1, 4),
            "roc_auc": round(test_roc, 4),
            "confusion_matrix": cm,
            "classification_report": report
        }
    }

    meta_file = os.path.join(output_dir, "transformer_metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    train_transformer_model()
