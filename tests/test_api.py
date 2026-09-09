"""
Integration tests for FastAPI endpoints.
"""

from fastapi.testclient import TestClient
import pytest

from src.serve import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify health check returns valid JSON status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_ui_endpoint(client):
    """Verify root / returns HTML web dashboard."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "VERTEX" in resp.text


def test_analyze_rejects_ssrf(client):
    """Verify /analyze endpoint blocks loopback SSRF."""
    resp = client.post("/analyze", json={"url": "http://127.0.0.1:8000/secret"})
    assert resp.status_code == 400
    assert "Security violation" in resp.json()["detail"]


def test_predict_single_image_upload(client, sample_image_bytes):
    """Verify single image upload to /predict."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "test.jpg"
    assert data["label"] in ["REAL", "AI_GENERATED"]
    assert "ai_probability" in data
    assert "real_probability" in data
    assert "confidence" in data
    assert "fft_spectrum" in data
    assert "fft_heatmap" in data
    assert "latency_ms" in data


def test_predict_batch_images(client, sample_image_bytes):
    """Verify batch multi-image upload endpoint."""
    files = [
        ("files", ("img1.jpg", sample_image_bytes, "image/jpeg")),
        ("files", ("img2.jpg", sample_image_bytes, "image/jpeg")),
    ]
    resp = client.post("/predict-batch", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert "label" in data["results"][0]
    assert "fft_heatmap" in data["results"][0]


def test_monitoring_status_endpoint(client):
    """Verify /monitoring/status returns drift statistics."""
    resp = client.get("/monitoring/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "drift_detected" in data
    assert "mean_wasserstein_distance" in data


def test_drift_report_html_endpoint(client):
    """Verify /reports/drift serves HTML report."""
    resp = client.get("/reports/drift")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert len(resp.text) > 100
