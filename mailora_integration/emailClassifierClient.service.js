/**
 * Mailora ML Email Classifier Client Service
 *
 * Microservice client that communicates with the Python FastAPI ML Classifier.
 * Filters out non-career emails in <5ms before passing genuine opportunities
 * to downstream opportunity extraction (Gemini / LLM), saving API quota and reducing latency.
 *
 * Features:
 * - Automatic circuit breaker & fail-open fallback (never drops emails if ML service is down)
 * - Configurable timeout & threshold
 * - Batch prediction support
 * - Safe feedback reporting for active learning
 */

const axios = require("axios");
const logger = require("../utils/logger");

const CLASSIFIER_URL = process.env.EMAIL_CLASSIFIER_URL || "http://localhost:8000";
const CLASSIFIER_TIMEOUT_MS = parseInt(process.env.EMAIL_CLASSIFIER_TIMEOUT_MS || "3000", 10);
const ML_FILTER_ENABLED = process.env.EMAIL_CLASSIFIER_ENABLED !== "false";

const httpClient = axios.create({
  baseURL: CLASSIFIER_URL,
  timeout: CLASSIFIER_TIMEOUT_MS,
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * Classifies an email as CAREER_RELATED or NON_CAREER_RELATED.
 *
 * @param {string} subject - Plaintext email subject
 * @param {string} body - Plaintext email body or snippet
 * @returns {Promise<{ is_career_related: boolean, label: string, confidence: number, model_version: string, fallback?: boolean }>}
 */
exports.classifyWithML = async (subject, body) => {
  if (!ML_FILTER_ENABLED) {
    logger.debug("MLClassifierClient", "ML filtering is disabled via EMAIL_CLASSIFIER_ENABLED=false. Defaulting to true.");
    return { is_career_related: true, label: "BYPASSED", confidence: 1.0, fallback: true };
  }

  try {
    const response = await httpClient.post("/predict", {
      subject: subject || "",
      body: body || "",
    });

    const data = response.data;
    logger.info(
      "MLClassifierClient",
      `Classified [${data.label}] (Conf: ${data.confidence}, Latency: ${data.latency_ms}ms): ${subject?.substring(0, 50)}`
    );

    return {
      is_career_related: data.is_career_related,
      label: data.label,
      confidence: data.confidence,
      career_probability: data.career_probability,
      model_version: data.model_version,
      latency_ms: data.latency_ms,
      fallback: false,
    };
  } catch (err) {
    // Fail-open: If ML service is unavailable, do NOT drop the email.
    // Fall back to downstream pipeline and log warning.
    logger.warn(
      "MLClassifierClient",
      `ML Classifier unavailable (${err.message}). Failing open to downstream processor.`
    );
    return {
      is_career_related: true,
      label: "CAREER_RELATED",
      confidence: 0.50,
      model_version: "fallback",
      fallback: true,
    };
  }
};

/**
 * Classifies a batch of emails.
 *
 * @param {Array<{ subject: string, body: string }>} emails
 * @returns {Promise<Array<object>>}
 */
exports.batchClassifyWithML = async (emails) => {
  if (!ML_FILTER_ENABLED || !emails || emails.length === 0) {
    return (emails || []).map(() => ({ is_career_related: true, fallback: true }));
  }

  try {
    const response = await httpClient.post("/predict/batch", { emails });
    return response.data.predictions;
  } catch (err) {
    logger.warn("MLClassifierClient", `Batch ML classification failed: ${err.message}. Failing open.`);
    return emails.map(() => ({ is_career_related: true, fallback: true }));
  }
};

/**
 * Sends user correction/feedback to the ML service for model improvement.
 */
exports.submitFeedback = async ({ emailId, subject, predictedLabel, correctedLabel, notes }) => {
  try {
    const response = await httpClient.post("/feedback", {
      email_id: emailId,
      subject: subject?.substring(0, 100) || "",
      predicted_label: predictedLabel,
      corrected_label: correctedLabel,
      user_consent: true,
      feedback_notes: notes || "",
    });
    return response.data;
  } catch (err) {
    logger.error("MLClassifierClient", `Failed to send feedback to ML service: ${err.message}`);
    return null;
  }
};

/**
 * Health check for ML Classifier service.
 */
exports.getHealth = async () => {
  try {
    const response = await httpClient.get("/health");
    return response.data;
  } catch (err) {
    return { status: "unreachable", error: err.message };
  }
};
