# Mailora ML Email Classifier

> **Production-grade Machine Learning & NLP Service for Binary Email Classification (`CAREER_RELATED` vs `NON_CAREER_RELATED`)**

Built as a dedicated, high-performance microservice for **Mailora**, the automated career opportunity detection and tracking system.

---

## 1. System Architecture

The classifier acts as a fast, intelligent gatekeeper sitting between Mailora's inbox ingestion engine and downstream opportunity extraction. By filtering out personal emails, shopping receipts, bank statements, social media alerts, and marketing/educational newsletters in under 5ms, it reduces external LLM API costs by 85–95% and eliminates rate-limit bottlenecks.

```
                  ┌──────────────────────────────────────────────┐
                  │          Incoming Email Ingestion            │
                  │        (Gmail Pub/Sub / Microsoft)           │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                            ┌───────────────────────────┐
                            │   Kafka Consumer Service  │
                            └─────────────┬─────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │      Stage 1: Fast ML Gatekeeper        │
                    │      (POST /predict - Python FastAPI)   │
                    │      is_career_related == true ?        │
                    └────────────┬─────────────────┬──────────┘
                                 │                 │
                       YES (1)   │                 │  NO (0)
                                 ▼                 ▼
         ┌────────────────────────────────┐   ┌───────────────────────────────┐
         │ Stage 2: Deep Extraction       │   │ Instant Discard / Archive     │
         │ (Gemini: Role, Company, Date,  │   │ • 0 external LLM tokens       │
         │  Links, Stage Classification)  │   │ • Sub-5ms response time       │
         │ • Save Opportunity in MongoDB  │   │ • Prevents noisy DB clutter   │
         │ • Schedule WhatsApp Reminders  │   └───────────────────────────────┘
         └────────────────────────────────┘
```

---

## 2. Model v1.1.0: False Positive Resolution & Contextual Disambiguation

### Root Cause Analysis (v1.0.0)
In v1.0.0, general newsletters and educational articles (such as the Glasp newsletter: *"Why AI hasn't replaced software engineers, and won't"*) were incorrectly classified as `CAREER_RELATED`. This was caused by two primary factors:
1. **Unigram Keyword Bias**: The initial seed dataset lacked hard negative examples discussing software engineering, AI, and careers in an educational or industry-report context. Single tokens like *"software"*, *"engineers"*, and *"ai"* had strong positive weights.
2. **Aggressive Recall Tuning**: The v1.0.0 threshold calibration tuned the decision boundary down to `0.30` to achieve 100% recall on the initial set, causing informational emails scoring ~0.35–0.45 to flip to false positives.

### Solution Architecture (v1.1.0)
- **Hard Negative Dataset Expansion**: Integrated 35+ diverse hard negative emails containing career vocabulary without actionable opportunities:
  - Educational newsletters (e.g., Glasp, TLDR AI, The Pragmatic Engineer).
  - Interview preparation tutorials and coding roadmaps (without interview invites).
  - Compensation and salary benchmark reports (without job offers).
  - University placement statistics summaries (without placement drive registrations).
  - Macro-economic hiring and layoff news reports (without recruitment calls).
- **High-Order Intent N-Grams**: Expanded baseline feature union to word trigrams (`ngram_range=(1, 3)`), enabling the model to learn the difference between actionable recruitment phrases (*"apply now at"*, *"interview scheduled for"*, *"online assessment invitation"*) and informational/reflective phrasing (*"why ai hasn't"*, *"top companies for"*, *"salary report for"*).
- **Calibrated Decision Boundary**: Re-calibrated decision threshold on the stratified validation set to `0.550` (Baseline) and `0.50` (Transformer), eliminating false alarms while preserving 97.3%–100% recall on genuine opportunities.

---

## 3. Model Implementations

### Stage A: Baseline Model (TF-IDF + Logistic Regression v1.1.0)
- **Feature Extraction**: Dual `FeatureUnion` combining:
  - Word-level n-grams (1–3) with sublinear TF scaling to capture both unigrams and complex action/informational phrases.
  - Character n-grams (3–5) within word boundaries to capture acronyms (*SDE*, *GET*, *CTC*, *PNR*, *EMI*, *OTP*) and spelling variants.
- **Classifier**: Logistic Regression with balanced class weighting (`class_weight="balanced"`), `C=1.5`, solver `liblinear`.
- **Validation**: 5-Fold Stratified Cross-Validation on training data.
- **Threshold Calibration**: Decision threshold calibrated on the validation set at `0.550` to balance precision and recall.
- **Artifact**: `artifacts/baseline_pipeline.joblib` (0.39 MB).

### Stage B: Advanced Transformer Model (DistilBERT v1.1.0)
- **Base Architecture**: `distilbert-base-uncased` with sequence classification head.
- **Sequence Handling**: Preserves critical opening context and tail application links/deadlines up to 256 tokens.
- **Optimization**: AdamW optimizer (`lr=2e-5`, `weight_decay=0.01`) with linear warmup scheduler.
- **Device Support**: Automatic device resolution supporting CPU and CUDA GPU acceleration.
- **Memory Footprint**: ~1.5 GB RAM during training; ~256 MB disk storage.
- **Artifacts**: `artifacts/distilbert/` (safetensors weights, config, and tokenizer).

---

## 4. Evaluation & Benchmarking Results

Both models were evaluated on the **same held-out, independent test set** (60 samples) with zero train/test data leakage.

| Metric | Stage A: Baseline (TF-IDF + LogReg v1.1.0) | Stage B: Advanced (DistilBERT v1.1.0) |
| :--- | :---: | :---: |
| **Accuracy** | **98.33%** | **100.00%** |
| **Precision** | **100.00%** | **100.00%** |
| **Recall (Career Opportunities)** | **97.30%** (36/37) | **100.00%** (37/37) |
| **F1-Score** | **0.9863** | **1.0000** |
| **ROC-AUC** | **1.0000** | **1.0000** |
| **False Positive Rate** | **0.00%** (0 false alarms) | **0.00%** (0 false alarms) |
| **False Positives** | **0** | **0** |
| **False Negatives** | **1** | **0** |
| **Hard Negative Subset (Newsletters & Trends)** | **7/7 Passed (100.0%)** | **7/7 Passed (100.0%)** |
| **Edge-Case Benchmark (15 cases)** | **15/15 Passed (100.0%)** | **15/15 Passed (100.0%)** |
| **Inference Latency** | **~4.5ms** | **~90ms** (CPU inference) |
| **Model Size** | **0.39 MB** | **256.11 MB** |

## 4. Project Structure

```
Model/
├── dataset/
│   ├── emails.csv            # Master cleaned, deduplicated dataset
│   ├── train.csv             # Stratified training split (70%)
│   ├── val.csv               # Validation split (15%) for threshold tuning
│   └── test.csv              # Independent test split (15%)
├── artifacts/
│   ├── baseline_pipeline.joblib  # Serialized scikit-learn pipeline
│   ├── model_metadata.json       # Baseline metrics, threshold & config
│   ├── evaluation_baseline.json  # Full baseline evaluation report
│   ├── evaluation_transformer.json # Full DistilBERT evaluation report
│   └── distilbert/               # Fine-tuned Transformer weights & tokenizer
│       ├── model.safetensors
│       ├── config.json
│       ├── tokenizer.json
│       └── transformer_metadata.json
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Text cleaning, normalization, subject-body fusion
│   ├── data_preparation.py  # Dataset generator, deduplication, stratified split
│   ├── train_baseline.py     # Stage A training & CV pipeline
│   ├── train_transformer.py  # Stage B DistilBERT fine-tuning
│   ├── predict.py            # Unified EmailPredictor engine (single & batch)
│   ├── evaluate.py           # Benchmarking & edge-case test suite
│   └── api.py                # FastAPI production service
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py    # Preprocessing & model unit/integration tests
│   └── test_api.py           # FastAPI endpoint integration tests
├── mailora_integration/
│   ├── emailClassifierClient.service.js  # Node.js client service for Mailora
│   └── integration_guide.md              # Step-by-step integration guide
├── .env.example              # Environment configuration template
├── .env                      # Local environment configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

## 5. Quickstart & Local Setup

### Step 1: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2: Prepare Dataset
```bash
python -m src.data_preparation
```

### Step 3: Train Models
```bash
# Train Stage A: Baseline Model (Fast, sub-second)
python -m src.train_baseline

# (Optional) Train Stage B: DistilBERT Transformer
python -m src.train_transformer
```

### Step 4: Run Evaluation & Benchmarks
```bash
# Evaluate baseline
python -m src.evaluate

# Evaluate transformer
python -c "from src.evaluate import run_evaluation; run_evaluation(model_type='transformer')"
```

### Step 5: Run Automated Tests
```bash
python -m pytest tests/ -v
```

### Step 6: Start the FastAPI Service
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation is interactively available at: **http://localhost:8000/docs**

### Step 7: Start the Responsive Web Frontend
```bash
cd frontend
npm install
npm run dev
```
Open **http://localhost:5173** in any web browser to test the interactive UI.

---

## 6. API Reference

### 1. `GET /health`
Returns service status, loaded model type, version, and threshold.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "baseline",
  "model_version": "1.0.0",
  "threshold": 0.30,
  "uptime_seconds": 45.2
}
```

### 2. `POST /predict`
Classifies a single email.

**Request:**
```json
{
  "subject": "Software Engineer Internship - Applications Open",
  "body": "We are inviting applications for our summer internship program at Atlassian..."
}
```

**Response:**
```json
{
  "label": "CAREER_RELATED",
  "is_career_related": true,
  "confidence": 0.9412,
  "career_probability": 0.9412,
  "model_version": "1.0.0",
  "latency_ms": 4.12
}
```

*Confidence calculation*: Directly derived from the posterior distribution. When classified as career, `confidence = career_probability`. When classified as non-career, `confidence = 1.0 - career_probability`.

### 3. `POST /predict/batch`
Classifies up to 100 emails in a single request with high throughput.

**Request:**
```json
{
  "emails": [
    { "subject": "Job Offer from Oracle", "body": "We are pleased to extend an offer for SDE II..." },
    { "subject": "Your Amazon order has shipped", "body": "Order #402 has been dispatched via BlueDart..." }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    { "label": "CAREER_RELATED", "is_career_related": true, "confidence": 0.965, "model_version": "1.0.0", "latency_ms": 2.1 },
    { "label": "NON_CAREER_RELATED", "is_career_related": false, "confidence": 0.942, "model_version": "1.0.0", "latency_ms": 2.1 }
  ],
  "total_emails": 2,
  "total_latency_ms": 4.2
}
```

### 4. `POST /feedback`
Consent-based active learning loop. Anonymizes the email reference and stores corrections in `artifacts/user_feedback.jsonl` without logging sensitive message bodies.

**Request:**
```json
{
  "email_id": "em_98124a",
  "subject": "Webinar on Generative AI Trends",
  "predicted_label": "CAREER_RELATED",
  "corrected_label": "NON_CAREER_RELATED",
  "user_consent": true,
  "feedback_notes": "General informational webinar, no active hiring."
}
```

---

## 7. Mailora Backend Integration

The client service has already been placed in Mailora's backend at:
`C:\Users\srini\Desktop\Mailora\mail-or-a-frontend-backend\server\services\emailClassifierClient.service.js`

To activate ML filtering in Mailora:

1. In `server/.env`, set:
   ```env
   EMAIL_CLASSIFIER_URL=http://localhost:8000
   EMAIL_CLASSIFIER_ENABLED=true
   EMAIL_CLASSIFIER_TIMEOUT_MS=3000
   ```

2. In `server/services/kafka/emailClassification.consumer.js`, import and call `classifyWithML`:
   ```javascript
   const { classifyWithML } = require("../emailClassifierClient.service");

   // Inside processEmailMessage():
   const mlResult = await classifyWithML(subject, snippet || body);
   if (!mlResult.is_career_related) {
     logger.info("KafkaConsumer", `[ML Filter] Skipped non-career email: ${subject?.substring(0, 50)}`);
     return; // Discards junk immediately — saves Gemini tokens!
   }

   // Genuine career emails continue to Gemini for stage & deadline extraction
   const aiResult = await classifyEmail(subject, snippet);
   ```

3. **Fail-Open Guarantee**: If the ML service is temporarily down, `emailClassifierClient` automatically fails open (`fallback: true`), passing all emails to Gemini so no email is ever lost.

---

## 8. Security & Privacy Safeguards

- **No Hardcoded Credentials**: All settings configured via `.env`.
- **Zero Content Leakage**: Raw email bodies and personal data are strictly excluded from API logs and error traces.
- **Payload Limits**: Built-in Pydantic validators reject oversized payloads (`MAX_SUBJECT_LENGTH=500`, `MAX_BODY_LENGTH=50000`).
- **Data Privacy**: Anonymized SHA-256 hash references used for feedback tracking.
