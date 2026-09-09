"""
Pytest configuration and test fixtures.
"""

import io
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_pil_image():
    """Generates a simple 64x64 synthetic RGB test image."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_image_bytes(sample_pil_image):
    """Encodes sample image as JPEG bytes."""
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def dummy_embeddings():
    """Returns random normalized embeddings simulating CLIP feature vectors."""
    n_samples = 20
    embed_dim = 512
    vecs = np.random.randn(n_samples, embed_dim).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    labels = np.random.choice([0, 1], size=n_samples)
    return vecs, labels
