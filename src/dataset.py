"""
CIFAKE dataset loader with stratified sampling and embedding cache management.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from src.config import settings
from src.utils import logger


# CIFAKE label mapping
LABEL_NAMES = {0: "REAL", 1: "AI_GENERATED"}


class CIFAKEDatasetLoader:
    """
    Manages CIFAKE dataset loading, stratified subsampling, and disk caching.
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        random_seed: int = 42,
    ):
        self.dataset_name = dataset_name or settings.dataset_name
        self.revision = revision or settings.dataset_revision
        self.cache_dir = cache_dir or settings.data_dir
        self.random_seed = random_seed
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(
        self,
        sample_size: Optional[int] = None,
        split: str = "train",
    ) -> Tuple[List[Image.Image], np.ndarray]:
        """
        Loads images and binary labels (0=REAL, 1=AI_GENERATED).
        If Hugging Face is unreachable or in offline mode, generates
        a deterministic local synthetic validation samples.

        Args:
            sample_size: Total images to load. None or <= 0 loads full split.
            split: 'train' or 'test'.

        Returns:
            Tuple of (images_list, labels_array).
        """
        try:
            from datasets import load_dataset as hf_load_dataset

            logger.info(
                f"Loading '{self.dataset_name}' (split={split}, rev={self.revision}) from Hugging Face / local cache..."
            )
            ds = hf_load_dataset(
                self.dataset_name,
                split=split,
                cache_dir=str(self.cache_dir),
                revision=self.revision,
            )
            images = [item["image"] for item in ds]
            labels = np.array([int(item["label"]) for item in ds], dtype=np.int64)
            logger.info(f"Loaded {len(images)} images from Hugging Face dataset.")

        except Exception as err:
            logger.warning(
                f"Could not load Hugging Face dataset '{self.dataset_name}' ({err}). "
                "Generating deterministic local synthetic validation samples..."
            )
            return self._generate_synthetic_samples(sample_size or 200)

        if sample_size and 0 < sample_size < len(images):
            images, labels = self._stratified_subsample(images, labels, sample_size)
            logger.info(f"Subsampled to {len(images)} stratified examples (seed={self.random_seed}).")

        return images, labels

    def _stratified_subsample(
        self,
        images: List[Image.Image],
        labels: np.ndarray,
        sample_size: int,
    ) -> Tuple[List[Image.Image], np.ndarray]:
        """Performs balanced stratified subsampling across classes."""
        rng = np.random.default_rng(self.random_seed)
        idx_0 = np.where(labels == 0)[0]
        idx_1 = np.where(labels == 1)[0]

        half_n = sample_size // 2
        selected_0 = rng.choice(idx_0, size=min(half_n, len(idx_0)), replace=False)
        selected_1 = rng.choice(idx_1, size=min(half_n, len(idx_1)), replace=False)

        selected_indices = np.concatenate([selected_0, selected_1])
        rng.shuffle(selected_indices)

        sub_images = [images[i] for i in selected_indices]
        sub_labels = labels[selected_indices]
        return sub_images, sub_labels

    def _generate_synthetic_samples(
        self,
        n_samples: int,
    ) -> Tuple[List[Image.Image], np.ndarray]:
        """Generates synthetic bootstrap images for offline testing."""
        rng = np.random.default_rng(self.random_seed)
        images = []
        labels = []
        for i in range(n_samples):
            lbl = i % 2
            # Create synthetic patterns
            if lbl == 0:
                # 'Real' smooth gradient
                arr = np.linspace(20, 220, 32 * 32, dtype=np.uint8).reshape(32, 32)
                arr = np.stack([arr, np.flipud(arr), np.fliplr(arr)], axis=-1)
            else:
                # 'AI' high-frequency grid noise
                arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
            images.append(Image.fromarray(arr, mode="RGB"))
            labels.append(lbl)
        return images, np.array(labels, dtype=np.int64)

    def get_embedding_cache_path(self, split: str, sample_size: int) -> Path:
        """Returns the file path for cached embeddings."""
        return self.cache_dir / f"embeddings_{split}_n{sample_size}.npz"

    def save_cached_embeddings(
        self,
        split: str,
        sample_size: int,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Path:
        """Saves extracted embeddings and labels to .npz file."""
        cache_path = self.get_embedding_cache_path(split, sample_size)
        np.savez_compressed(cache_path, embeddings=embeddings, labels=labels)
        logger.info(f"Saved cached embeddings to {cache_path} ({embeddings.shape}).")
        return cache_path

    def load_cached_embeddings(
        self,
        split: str,
        sample_size: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Loads cached embeddings if present on disk."""
        cache_path = self.get_embedding_cache_path(split, sample_size)
        if cache_path.exists():
            logger.info(f"Found cached embeddings at {cache_path}.")
            data = np.load(cache_path)
            return data["embeddings"], data["labels"]
        return None
