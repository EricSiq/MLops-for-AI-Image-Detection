"""
Unit tests for drift detection and report generation.
"""

import numpy as np
import pytest

from src.monitor import compute_distribution_drift, generate_drift_report


def test_compute_distribution_drift_no_drift():
    """Verify identical distributions result in no detected drift."""
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=(100, 10))
    cur = rng.normal(loc=0.0, scale=1.0, size=(100, 10))

    metrics = compute_distribution_drift(ref, cur)
    assert not metrics["drift_detected"]
    assert metrics["drift_share"] <= 0.2
    assert metrics["mean_wasserstein_distance"] < 0.5


def test_compute_distribution_drift_with_shift():
    """Verify shifted distribution triggers drift detection."""
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=(100, 10))
    cur = rng.normal(loc=5.0, scale=1.0, size=(100, 10))  # Significant mean shift

    metrics = compute_distribution_drift(ref, cur)
    assert metrics["drift_detected"]
    assert metrics["drift_share"] > 0.5
    assert metrics["mean_wasserstein_distance"] > 3.0


def test_generate_drift_report_outputs(tmp_path):
    """Verify drift report files (HTML and JSON) are properly created."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, size=(50, 16))
    cur = rng.normal(0.5, 1, size=(50, 16))

    html_file = tmp_path / "drift.html"
    json_file = tmp_path / "drift.json"

    metrics = generate_drift_report(
        reference_embeddings=ref,
        current_embeddings=cur,
        output_html_path=html_file,
        output_json_path=json_file,
    )

    assert html_file.exists()
    assert json_file.exists()
    assert "drift_detected" in metrics
    assert html_file.stat().st_size > 100
    assert json_file.stat().st_size > 50
