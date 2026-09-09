"""
Model architecture module: Frozen CLIP Vision Backbone + Scikit-Learn Classifier Head.
Supports PyTorch GPU/CPU inference and ONNX runtime export.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from PIL import Image

from src.config import settings
from src.preprocess import load_image_safely, preprocess_image_numpy
from src.utils import logger


class CLIPFeatureExtractor:
    """
    Extracts frozen visual embeddings using CLIP ViT architecture.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        revision: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.clip_model_name
        self.revision = revision or settings.clip_revision
        self.device = self._determine_device(device)
        self._model = None
        self._processor = None

    def _determine_device(self, requested_device: Optional[str]) -> str:
        req = requested_device or settings.device
        if req == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    # Test actual kernel execution to catch architecture mismatches (e.g. sm_120 Blackwell)
                    try:
                        probe_t = torch.zeros((2, 2), device="cuda")
                        _ = torch.matmul(probe_t, probe_t)
                        return "cuda"
                    except Exception as probe_err:
                        logger.warning(
                            f"CUDA detected but kernel execution probe failed ({probe_err}). "
                            "Gracefully falling back to multi-threaded CPU execution."
                        )
                        return "cpu"
            except ImportError:
                pass
        return "cpu"

    def _load_model(self) -> None:
        """Lazy loads CLIP model and processor to minimize startup memory."""
        if self._model is not None:
            return

        import torch
        from transformers import CLIPProcessor, CLIPVisionModelWithProjection

        logger.info(
            f"Loading CLIP vision backbone '{self.model_name}' (rev='{self.revision}') on device='{self.device}'..."
        )
        self._processor = CLIPProcessor.from_pretrained(self.model_name, revision=self.revision)
        self._model = CLIPVisionModelWithProjection.from_pretrained(
            self.model_name, revision=self.revision
        )
        self._model.to(self.device)
        self._model.eval()

        # Freeze backbone parameters
        for param in self._model.parameters():
            param.requires_grad = False

    def extract_features_pil(
        self,
        images: List[Image.Image],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Extracts L2-normalized 512-dim visual embeddings from a list of PIL Images.
        """
        self._load_model()
        import torch

        all_embeddings = []
        n_images = len(images)

        for i in range(0, n_images, batch_size):
            batch_images = [load_image_safely(img) for img in images[i : i + batch_size]]
            inputs = self._processor(images=batch_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # image_embeds is already projected to embedding dimension (512)
                embeds = outputs.image_embeds
                # L2 normalize embeddings
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                all_embeddings.append(embeds.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def extract_single(self, image: Union[Image.Image, bytes, Path, str]) -> np.ndarray:
        """Extracts normalized embedding for a single image."""
        pil_img = load_image_safely(image)
        embeds = self.extract_features_pil([pil_img], batch_size=1)
        return embeds[0]


class AIImageClassifier:
    """
    Linear Probe / Classifier head trained on top of frozen CLIP embeddings.
    Provides calibrated probability estimates and serialization.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        from sklearn.linear_model import LogisticRegression

        self.C = C
        self.max_iter = max_iter
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=settings.random_seed,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AIImageClassifier":
        """Fits the linear classifier on extracted embeddings."""
        logger.info(f"Fitting LogisticRegression classifier on {X.shape[0]} samples (C={self.C})...")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary class (0=REAL, 1=AI_GENERATED)."""
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probabilities of shape (N, 2), where:
          column 0 = P(REAL)
          column 1 = P(AI_GENERATED)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted yet.")
        return self.model.predict_proba(X)

    def save(self, filepath: Union[str, Path]) -> Path:
        """Saves model artifact using joblib and writes SHA256 checksum for integrity verification."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)

        from src.utils import compute_file_sha256

        sha256_hash = compute_file_sha256(filepath)
        sha_file = filepath.with_suffix(filepath.suffix + ".sha256")
        with open(sha_file, "w", encoding="utf-8") as f:
            f.write(sha256_hash)

        logger.info(f"Model successfully saved to {filepath} (SHA256: {sha256_hash[:12]}...).")
        return filepath

    def load(
        self, filepath: Union[str, Path], verify_checksum: bool = True
    ) -> "AIImageClassifier":
        """Loads model artifact from disk with cryptographic checksum verification."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")

        if verify_checksum:
            sha_file = filepath.with_suffix(filepath.suffix + ".sha256")
            if sha_file.exists():
                from src.utils import compute_file_sha256

                expected_hash = sha_file.read_text(encoding="utf-8").strip()
                actual_hash = compute_file_sha256(filepath)
                if expected_hash != actual_hash:
                    raise ValueError(
                        f"Model integrity verification failed! Expected {expected_hash}, got {actual_hash}"
                    )
                logger.info("Model SHA256 integrity check passed.")

        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded classifier from {filepath}.")
        return self
