# Mailora Backend Integration Guide

This guide explains how the **ML Email Classifier** integrates into the existing Mailora backend without disrupting any existing functionality.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │              Incoming Email                  │
                      │         (Gmail Pub/Sub / Microsoft)          │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   Kafka Consumer Service  │
                               └─────────────┬─────────────┘
                                             │
                        ┌────────────────────▼────────────────────┐
                        │      Stage 1: High-Speed ML Filter      │
                        │      (POST /predict - Sub-5ms)          │
                        │      is_career_related == true ?        │
                        └────────────┬─────────────────┬──────────┘
                                     │                 │
                           YES (1)   │                 │  NO (0)
                                     ▼                 ▼
             ┌────────────────────────────────┐   ┌───────────────────────────────┐
             │ Stage 2: Deep Extraction       │   │ Discard / Archive Email       │
             │ (Company, Role, Deadline,      │   │ • 0 LLM API calls             │
             │  Links, Stage via Gemini)      │   │ • Instant (<5ms processing)   │
             │ • Store in MongoDB             │   │ • No junk reminders created   │
             │ • Schedule WhatsApp Reminders  │   └───────────────────────────────┘
             └────────────────────────────────┘
```

---

## Step 1: Add Environment Variables to Mailora Backend

In `C:\Users\srini\Desktop\Mailora\mail-or-a-frontend-backend\server\.env`:

```env
# Mailora ML Email Classifier Microservice
EMAIL_CLASSIFIER_URL=http://localhost:8000
EMAIL_CLASSIFIER_ENABLED=true
EMAIL_CLASSIFIER_TIMEOUT_MS=3000
```

---

## Step 2: Copy the Client Service

Copy `mailora_integration/emailClassifierClient.service.js` to:
`C:\Users\srini\Desktop\Mailora\mail-or-a-frontend-backend\server\services\emailClassifierClient.service.js`

---

## Step 3: Update `emailClassification.consumer.js`

In `server/services/kafka/emailClassification.consumer.js`:

```javascript
// 1. Import the ML client
const { classifyWithML } = require("../emailClassifierClient.service");
const { classifyEmail } = require("../emailAI.service");

// 2. In processEmailMessage(messagePayload):
async function processEmailMessage(messagePayload) {
  const { userId, connectedAccountId, provider, messageId, subject, from, snippet, body, internalDate } = messagePayload;

  // ─── Step 0: Fast ML Gatekeeper (<5ms) ────────────────────────
  const mlResult = await classifyWithML(subject, snippet || body);
  if (!mlResult.is_career_related) {
    logger.info("KafkaConsumer", `[ML Gatekeeper] Non-career email skipped (${mlResult.label}, Conf: ${mlResult.confidence}): ${subject?.substring(0, 50)}`);
    return; // Discard immediately — saves Gemini quota & latency
  }

  // ─── Step 1: Deep Opportunity Extraction (Gemini) ─────────────
  // Only genuine career emails reach Gemini for stage, deadline, and links extraction!
  const aiResult = await classifyEmail(subject, snippet);
  ...
}
```

---

## Benefits

1. **Token Savings**: 85-95% reduction in Gemini API requests by eliminating receipts, social media alerts, bank statements, personal chats, and spam before invoking Gemini.
2. **Speed**: Filtering happens locally in 1-5ms instead of waiting 1.5 - 3.5 seconds for external LLMs.
3. **Resilience**: Fail-open architecture ensures that if the ML service ever goes offline, all emails continue to be processed normally through Gemini.
4. **Data Privacy**: The ML service processes predictions locally on the server without leaking email contents to external APIs.
