"""
Unit tests for AI Image Classifier head and feature extractor.
"""

import numpy as np
import pytest

from src.model import AIImageClassifier


def test_classifier_fit_and_predict(dummy_embeddings):
    """Verify fitting, predicting labels, and predicting calibrated probabilities."""
    X, y = dummy_embeddings
    clf = AIImageClassifier(C=1.0)
    clf.fit(X, y)

    assert clf.is_fitted

    # Predict labels
    preds = clf.predict(X)
    assert preds.shape == (len(y),)
    assert set(np.unique(preds)).issubset({0, 1})

    # Predict probabilities
    probs = clf.predict_proba(X)
    assert probs.shape == (len(y), 2)
    # Check probability axioms
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_classifier_save_and_load(tmp_path, dummy_embeddings):
    """Verify serialization roundtrip of classifier using joblib."""
    X, y = dummy_embeddings
    clf = AIImageClassifier(C=0.5)
    clf.fit(X, y)

    save_file = tmp_path / "model.joblib"
    clf.save(save_file)
    assert save_file.exists()

    loaded_clf = AIImageClassifier().load(save_file)
    assert loaded_clf.is_fitted

    preds_original = clf.predict(X)
    preds_loaded = loaded_clf.predict(X)
    np.testing.assert_array_equal(preds_original, preds_loaded)
