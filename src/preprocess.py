"""
Image preprocessing and frequency-domain artifact extraction module.
Designed as pure, deterministic, and modular functions suitable for both
training and inference environments.
"""

from io import BytesIO
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps


# Standard CLIP normalization parameters
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)

# Prevent decompression bomb attacks (50 megapixels max)
Image.MAX_IMAGE_PIXELS = 50_000_000


def load_image_safely(
    image_source: Union[str, bytes, BytesIO, Image.Image],
    min_size: Tuple[int, int] = (32, 32),
) -> Image.Image:
    """
    Safely loads and validates an image from bytes, path, or existing PIL object.
    Converts RGBA/grayscale/palette to standard 3-channel RGB.

    Args:
        image_source: File path, raw bytes, BytesIO buffer, or PIL Image.
        min_size: Minimum allowed (width, height).

    Returns:
        Validated PIL Image in RGB format.

    Raises:
        ValueError: If image fails verification or does not meet minimum dimensions.
    """
    if isinstance(image_source, Image.Image):
        img = image_source
    elif isinstance(image_source, (bytes, bytearray)):
        img = Image.open(BytesIO(image_source))
    elif isinstance(image_source, (str, BytesIO)):
        img = Image.open(image_source)
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    # Verify and load pixels
    img.load()

    # Dimension check
    width, height = img.size
    if width < min_size[0] or height < min_size[1]:
        raise ValueError(
            f"Image dimensions ({width}x{height}) are smaller than minimum allowed {min_size}"
        )

    # Handle EXIF orientation if present
    img = ImageOps.exif_transpose(img)

    # Standardize to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def resize_and_crop_clip(image: Image.Image, target_size: int = 224) -> Image.Image:
    """
    Resizes image maintaining aspect ratio and center crops to target_size x target_size.
    Uses BICUBIC resampling matching CLIP specification.
    """
    w, h = image.size
    scale = target_size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Resize shortest edge to target_size
    resized = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size

    return resized.crop((left, top, right, bottom))


def preprocess_image_numpy(
    image: Image.Image,
    target_size: int = 224,
) -> np.ndarray:
    """
    Pure NumPy preprocessing pipeline for CLIP image encoder.
    Outputs normalized float32 array with shape (3, target_size, target_size).
    """
    cropped = resize_and_crop_clip(image, target_size=target_size)
    arr = np.asarray(cropped, dtype=np.float32) / 255.0  # (H, W, C) in [0, 1]

    # Transpose to (C, H, W)
    arr = np.transpose(arr, (2, 0, 1))

    # Standardize with CLIP mean and std
    normalized = (arr - CLIP_MEAN) / CLIP_STD
    return normalized.astype(np.float32)


def compute_radial_fft_spectrum(
    image: Image.Image,
    num_bins: int = 32,
    size: int = 128,
) -> np.ndarray:
    """
    Computes azimuthally averaged radial power spectrum via 2D FFT.
    Useful for identifying high-frequency grid and deconvolution artifacts
    characteristic of synthetic generative models (GANs, Diffusion).

    Args:
        image: Input PIL Image.
        num_bins: Number of radial frequency bins.
        size: Target square size for FFT analysis.

    Returns:
        Normalized 1D radial power spectrum feature vector of length num_bins.
    """
    gray = image.convert("L").resize((size, size), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(gray, dtype=np.float32) / 255.0

    # 2D Fast Fourier Transform
    fft2 = np.fft.fft2(arr)
    fft_shifted = np.fft.fftshift(fft2)
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1e-7)

    # Compute radial distance grid from center
    center = size // 2
    y, x = np.ogrid[:size, :size]
    radial_dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    max_radius = np.sqrt(2 * (center ** 2))
    bin_edges = np.linspace(0, max_radius, num_bins + 1)

    profile = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        mask = (radial_dist >= bin_edges[i]) & (radial_dist < bin_edges[i + 1])
        if np.any(mask):
            profile[i] = np.mean(magnitude_spectrum[mask])

    # Normalize profile to zero-mean unit-variance
    norm_profile = (profile - np.mean(profile)) / (np.std(profile) + 1e-7)
    return norm_profile.astype(np.float32)
