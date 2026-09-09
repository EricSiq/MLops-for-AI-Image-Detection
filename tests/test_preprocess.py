"""
Unit tests for image preprocessing and FFT frequency extraction.
"""

import io
import numpy as np
import pytest
from PIL import Image

from src.preprocess import (
    compute_radial_fft_spectrum,
    load_image_safely,
    preprocess_image_numpy,
    resize_and_crop_clip,
)


def test_load_image_safely_from_pil(sample_pil_image):
    """Test loading directly from PIL Image."""
    loaded = load_image_safely(sample_pil_image)
    assert loaded.mode == "RGB"
    assert loaded.size == (64, 64)


def test_load_image_safely_from_bytes(sample_image_bytes):
    """Test loading from raw JPEG bytes."""
    loaded = load_image_safely(sample_image_bytes)
    assert loaded.mode == "RGB"
    assert loaded.size == (64, 64)


def test_load_image_safely_rejects_undersized():
    """Test that images smaller than min_size are rejected."""
    tiny = Image.new("RGB", (16, 16))
    buf = io.BytesIO()
    tiny.save(buf, format="PNG")
    with pytest.raises(ValueError, match="smaller than minimum allowed"):
        load_image_safely(buf.getvalue(), min_size=(32, 32))


def test_load_image_safely_handles_rgba():
    """Test that RGBA images are converted to 3-channel RGB."""
    rgba = Image.new("RGBA", (48, 48), color=(255, 0, 0, 128))
    loaded = load_image_safely(rgba)
    assert loaded.mode == "RGB"
    assert loaded.size == (48, 48)


def test_resize_and_crop_clip(sample_pil_image):
    """Test that resize_and_crop_clip yields exact target square dimensions."""
    cropped = resize_and_crop_clip(sample_pil_image, target_size=224)
    assert cropped.size == (224, 224)


def test_preprocess_image_numpy(sample_pil_image):
    """Test NumPy preprocessing produces normalized (3, 224, 224) float32 array."""
    arr = preprocess_image_numpy(sample_pil_image, target_size=224)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32
    # Check that normalization produces reasonable range
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_compute_radial_fft_spectrum(sample_pil_image):
    """Test 2D FFT radial spectrum feature extraction."""
    spectrum = compute_radial_fft_spectrum(sample_pil_image, num_bins=32)
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.shape == (32,)
    assert spectrum.dtype == np.float32
    assert not np.isnan(spectrum).any()
