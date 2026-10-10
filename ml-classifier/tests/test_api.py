"""
Integration Tests for Mailora FastAPI Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"healthy", "degraded"}
    assert data["model_loaded"] is True
    assert "model_version" in data
    assert "uptime_seconds" in data
    assert 0.0 <= data["threshold"] <= 1.0


def test_predict_career_email(client):
    payload = {
        "subject": "Software Engineer Internship - Applications Open",
        "body": "We are inviting applications for our summer engineering internship program. Apply before Nov 15."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "CAREER_RELATED"
    assert data["is_career_related"] is True
    assert 0.5 <= data["confidence"] <= 1.0
    assert 0.0 <= data["career_probability"] <= 1.0
    assert "model_version" in data
    assert "latency_ms" in data


def test_predict_non_career_email(client):
    payload = {
        "subject": "Your Netflix monthly subscription has renewed",
        "body": "Your payment of INR 649 was successfully charged for your Netflix Premium plan."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NON_CAREER_RELATED"
    assert data["is_career_related"] is False
    assert 0.5 <= data["confidence"] <= 1.0


def test_predict_empty_content_returns_400(client):
    payload = {"subject": "   ", "body": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_batch_endpoint(client):
    payload = {
        "emails": [
            {
                "subject": "Hiring: Lead Backend Developer at Atlassian",
                "body": "Apply for the Senior Backend Engineer role in Bangalore."
            },
            {
                "subject": "Zomato Receipt: Order confirmed",
                "body": "Your payment of INR 450 was received. Food is on its way."
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_emails"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["is_career_related"] is True
    assert data["predictions"][1]["is_career_related"] is False
    assert data["total_latency_ms"] >= 0.0


def test_feedback_with_consent(client):
    payload = {
        "email_id": "test_email_9981",
        "subject": "Join our Python coding webinar",
        "predicted_label": "CAREER_RELATED",
        "corrected_label": "NON_CAREER_RELATED",
        "user_consent": True,
        "feedback_notes": "This was an educational marketing webinar, not a job drive."
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"recorded", "disabled"}
    assert "recorded_at" in data


def test_feedback_without_consent(client):
    payload = {
        "email_id": "test_email_1234",
        "predicted_label": "NON_CAREER_RELATED",
        "corrected_label": "CAREER_RELATED",
        "user_consent": False
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "skipped"


def test_feedback_invalid_label_returns_422(client):
    payload = {
        "predicted_label": "INVALID_LABEL",
        "corrected_label": "CAREER_RELATED",
        "user_consent": True
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 422
