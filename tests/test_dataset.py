"""
Unit tests for CIFAKE dataset loader and caching mechanism.
"""

import numpy as np
import pytest
from PIL import Image

from src.dataset import CIFAKEDatasetLoader


def test_dataset_synthetic_fallback(tmp_path):
    """Verify that CIFAKEDatasetLoader produces balanced samples in fallback mode."""
    loader = CIFAKEDatasetLoader(dataset_name="non-existent/dataset", cache_dir=tmp_path)
    images, labels = loader.load_dataset(sample_size=20, split="train")

    assert len(images) == 20
    assert len(labels) == 20
    assert isinstance(images[0], Image.Image)
    assert set(np.unique(labels)).issubset({0, 1})
    # Check balance
    assert np.sum(labels == 0) == 10
    assert np.sum(labels == 1) == 10


def test_dataset_embedding_cache_roundtrip(tmp_path, dummy_embeddings):
    """Verify saving and loading cached embeddings from disk."""
    loader = CIFAKEDatasetLoader(cache_dir=tmp_path)
    embeddings, labels = dummy_embeddings

    # Initially not cached
    assert loader.load_cached_embeddings("train", 20) is None

    # Save to cache
    save_path = loader.save_cached_embeddings("train", 20, embeddings, labels)
    assert save_path.exists()

    # Load back
    loaded = loader.load_cached_embeddings("train", 20)
    assert loaded is not None
    loaded_embeds, loaded_labels = loaded

    np.testing.assert_allclose(embeddings, loaded_embeds)
    np.testing.assert_array_equal(labels, loaded_labels)
