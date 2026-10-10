"""
Unit and Integration Tests for Mailora Classifier Model & Preprocessing.
"""

import pytest
import os
from src.preprocess import clean_text, truncate_smart, combine_subject_and_body, text_dedup_key
from src.predict import EmailPredictor


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text Preprocessing Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_text_html_stripping():
    html_input = "<html><body><p>Hello <b>World</b>!</p><br/>Link: <a href='https://jobs.com'>Jobs</a></body></html>"
    cleaned = clean_text(html_input)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "Hello World !" in cleaned or "Hello World!" in cleaned


def test_clean_text_entity_decoding():
    raw = "Tech &amp; Engineering &quot;Role&quot; at Razorpay &gt; Bangalore"
    cleaned = clean_text(raw)
    assert "&amp;" not in cleaned
    assert "Tech & Engineering \"Role\" at Razorpay > Bangalore" in cleaned


def test_clean_text_url_and_email_normalization():
    raw = "Contact us at hiring@company.com or visit https://jobs.company.com/apply for details."
    cleaned = clean_text(raw)
    assert "EMAIL_ADDR" in cleaned
    assert "URL_LINK" in cleaned
    assert "hiring@company.com" not in cleaned


def test_truncate_smart_preserves_head_and_tail():
    long_body = "HEAD_START " + ("Middle text content " * 300) + " TAIL_END"
    truncated = truncate_smart(long_body, max_chars=400)
    assert len(truncated) <= 450
    assert "HEAD_START" in truncated
    assert "TAIL_END" in truncated
    assert "[TRUNCATED]" in truncated


def test_combine_subject_and_body_weighting():
    subject = "Urgent: Interview Invitation"
    body = "Your technical interview is scheduled for Friday."
    fused = combine_subject_and_body(subject, body, subject_weight=2)
    assert fused.count("Urgent: Interview Invitation") >= 2
    assert "Subject:" in fused
    assert "Body:" in fused


def test_text_dedup_key_near_duplicates():
    text1_sub = "Software Engineer Role at Google!"
    text1_bod = "Apply at https://careers.google.com/1"

    text2_sub = "software engineer role at google"
    text2_bod = "apply at https://careers.google.com/2"

    key1 = text_dedup_key(text1_sub, text1_bod)
    key2 = text_dedup_key(text2_sub, text2_bod)
    assert key1 == key2, "Canonical signatures should identify normalized near-duplicates."


# ─────────────────────────────────────────────────────────────────────────────
# 2. Classifier Prediction Tests (Baseline Model)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def predictor():
    pred = EmailPredictor(model_type="baseline", artifacts_dir="artifacts", threshold_override=0.45)
    assert pred.is_loaded, "Baseline model must be trained and saved in artifacts."
    return pred


def test_clear_positive_career_email(predictor):
    subject = "Amazon Campus Drive: SDE I Assessment Invitation"
    body = "Dear candidate, please complete the HackerRank coding challenge for the Software Development Engineer role at Amazon."
    res = predictor.predict_single(subject, body)
    assert res["label"] == "CAREER_RELATED"
    assert res["is_career_related"] is True
    assert 0.5 <= res["confidence"] <= 1.0
    assert res["latency_ms"] >= 0.0


def test_clear_negative_shopping_email(predictor):
    subject = "Your Swiggy order from Biryani Blues is arriving in 15 mins"
    body = "Track delivery partner Ramesh live on the Swiggy app. Total paid: INR 350 via UPI."
    res = predictor.predict_single(subject, body)
    assert res["label"] == "NON_CAREER_RELATED"
    assert res["is_career_related"] is False
    assert 0.5 <= res["confidence"] <= 1.0


def test_clear_negative_banking_email(predictor):
    subject = "HDFC Bank: Monthly Credit Card e-Statement"
    body = "Your credit card e-statement for billing period ending Oct 2026 is attached. Amount due: INR 12,450."
    res = predictor.predict_single(subject, body)
    assert res["label"] == "NON_CAREER_RELATED"
    assert res["is_career_related"] is False


def test_rejection_email_is_career_related(predictor):
    """
    Critical requirement: Rejection emails ARE career-related status updates!
    """
    subject = "Update regarding your application for Backend Engineer at Stripe"
    body = "Thank you for interviewing with us. After careful consideration, we have decided not to move forward with your application."
    res = predictor.predict_single(subject, body)
    assert res["label"] == "CAREER_RELATED"
    assert res["is_career_related"] is True


def test_short_email_handling(predictor):
    subject = "Interview at 4 PM"
    body = "Google Meet: https://meet.google.com/abc-xyz"
    res = predictor.predict_single(subject, body)
    assert res["label"] == "CAREER_RELATED"
    assert res["is_career_related"] is True


def test_empty_input_handling(predictor):
    res = predictor.predict_single("", "")
    assert res["label"] == "NON_CAREER_RELATED"
    assert res["is_career_related"] is False
    assert res["confidence"] == 0.50


def test_batch_prediction_consistency(predictor):
    emails = [
        {"subject": "Job Offer from Oracle", "body": "We are pleased to extend an employment offer for SDE II."},
        {"subject": "Your Amazon order shipped", "body": "Order #123 has been shipped via BlueDart."},
    ]
    batch_res = predictor.predict_batch(emails)
    assert len(batch_res) == 2

    # Verify predictions match single inferences
    single_res_0 = predictor.predict_single(emails[0]["subject"], emails[0]["body"])
    single_res_1 = predictor.predict_single(emails[1]["subject"], emails[1]["body"])

    assert batch_res[0]["label"] == single_res_0["label"]
    assert batch_res[1]["label"] == single_res_1["label"]
    assert abs(batch_res[0]["career_probability"] - single_res_0["career_probability"]) < 1e-4
