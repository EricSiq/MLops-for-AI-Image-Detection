"""
High-performance inference engine with ONNX export and runtime execution.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from PIL import Image

from src.config import settings
from src.model import AIImageClassifier, CLIPFeatureExtractor
from src.preprocess import compute_radial_fft_spectrum, load_image_safely
from src.utils import logger


def export_classifier_to_onnx(
    classifier: AIImageClassifier,
    output_path: Path,
) -> Path:
    """
    Exports trained LogisticRegression classifier head to ONNX format.
    Uses skl2onnx if installed, or creates a standardized ONNX graph.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, settings.embedding_dim]))]
        onx = convert_sklearn(classifier.model, initial_types=initial_type)
        with open(output_path, "wb") as f:
            f.write(onx.SerializeToString())
        logger.info(f"Exported ONNX classifier to {output_path} via skl2onnx.")
        return output_path
    except ImportError:
        pass

    # Fallback: create ONNX linear classifier using torch.onnx
    import torch
    import torch.nn as nn

    class LinearProbeONNX(nn.Module):
        def __init__(self, weight: np.ndarray, bias: np.ndarray):
            super().__init__()
            self.linear = nn.Linear(weight.shape[1], 2)
            # Binary logistic regression in sklearn has coef_ of shape (1, D)
            # We map it to 2 classes: class 0 (-z) and class 1 (+z)
            w2 = np.vstack([-weight, weight])  # (2, D)
            b2 = np.array([-bias[0], bias[0]], dtype=np.float32)  # (2,)
            self.linear.weight.data = torch.from_numpy(w2).float()
            self.linear.bias.data = torch.from_numpy(b2).float()

        def forward(self, x):
            logits = self.linear(x)
            probabilities = torch.softmax(logits, dim=-1)
            return probabilities

    model_onnx = LinearProbeONNX(classifier.model.coef_, classifier.model.intercept_)
    model_onnx.eval()

    dummy_input = torch.randn(1, settings.embedding_dim, dtype=torch.float32)
    torch.onnx.export(
        model_onnx,
        dummy_input,
        str(output_path),
        input_names=["embeddings"],
        output_names=["probabilities"],
        dynamic_axes={"embeddings": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        opset_version=14,
    )
    logger.info(f"Exported ONNX classifier to {output_path} via PyTorch ONNX.")
    return output_path


class AIImagePredictor:
    """
    Production-grade inference pipeline combining CLIP visual embedding extraction
    with ONNX / Scikit-Learn classifier head.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_onnx: bool = True,
    ):
        self.feature_extractor = CLIPFeatureExtractor()
        self.use_onnx = use_onnx
        self.onnx_session = None
        self.joblib_model = None

        # Resolve model path
        self.onnx_path = settings.onnx_model_path
        self.joblib_path = model_path or settings.joblib_model_path

        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initializes ONNX runtime if available and requested, otherwise loads joblib."""
        if self.use_onnx and self.onnx_path.exists():
            try:
                import onnxruntime as ort

                # Prefer CUDAExecutionProvider if available, otherwise CPU
                providers = ["CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")

                self.onnx_session = ort.InferenceSession(str(self.onnx_path), providers=providers)
                logger.info(f"Loaded ONNX model for inference with providers={providers}.")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX session ({e}). Falling back to joblib.")

        if self.joblib_path.exists():
            self.joblib_model = joblib.load(self.joblib_path)
            logger.info(f"Loaded joblib classifier from {self.joblib_path}.")
        else:
            logger.warning(
                f"No saved model found at {self.joblib_path} or {self.onnx_path}. "
                "Predictor will require fit or model save before inference."
            )

    def _predict_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Runs classifier head on precomputed embeddings.
        Returns array of shape (N, 2) where col 0 = P(Real), col 1 = P(AI).
        """
        if self.onnx_session is not None:
            input_name = self.onnx_session.get_inputs()[0].name
            raw_output = self.onnx_session.run(None, {input_name: embeddings.astype(np.float32)})
            probs = raw_output[0]
            # Handle dictionary or array outputs from different ONNX exporters
            if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], dict):
                probs = np.array([[p[0], p[1]] for p in probs], dtype=np.float32)
            elif probs.ndim == 1:
                probs = np.vstack([1.0 - probs, probs]).T
            return probs.astype(np.float32)

        elif self.joblib_model is not None:
            return self.joblib_model.predict_proba(embeddings).astype(np.float32)

        else:
            # Fallback heuristic baseline if no weights trained yet
            logger.warning("Using fallback baseline classifier.")
            n = embeddings.shape[0]
            # Neutral 0.5 baseline
            return np.full((n, 2), 0.5, dtype=np.float32)

    def predict_image(
        self,
        image_source: Union[Image.Image, bytes, Path, str],
        include_fft: bool = True,
    ) -> Dict[str, Any]:
        """
        Predicts whether a single image is Real or AI-generated.
        Includes latency telemetry and deep forensic frequency artifacts.
        """
        import time
        from src.preprocess import generate_fft_magnitude_heatmap_base64

        t0 = time.perf_counter()
        image = load_image_safely(image_source)
        t_load = time.perf_counter()

        embedding = self.feature_extractor.extract_features_pil([image], batch_size=1)
        t_embed = time.perf_counter()

        probs = self._predict_embeddings(embedding)[0]
        t_infer = time.perf_counter()

        real_prob = float(probs[0])
        ai_prob = float(probs[1])

        label = "AI_GENERATED" if ai_prob >= 0.5 else "REAL"
        confidence = max(real_prob, ai_prob)

        feature_latency_ms = round((t_embed - t_load) * 1000, 2)
        inference_latency_ms = round((t_infer - t_embed) * 1000, 2)
        total_latency_ms = round((t_infer - t0) * 1000, 2)

        result: Dict[str, Any] = {
            "label": label,
            "ai_probability": round(ai_prob, 4),
            "real_probability": round(real_prob, 4),
            "confidence": round(confidence, 4),
            "latency_ms": total_latency_ms,
            "feature_latency_ms": feature_latency_ms,
            "inference_latency_ms": inference_latency_ms,
        }

        if include_fft:
            spectrum = compute_radial_fft_spectrum(image, num_bins=32)
            result["fft_spectrum"] = [round(float(v), 4) for v in spectrum]
            result["fft_heatmap"] = generate_fft_magnitude_heatmap_base64(image, size=128)

        return result

    def predict_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
        include_fft: bool = False,
    ) -> List[Dict[str, Any]]:
        """Batched prediction for multiple PIL images."""
        if not images:
            return []

        import time
        from src.preprocess import generate_fft_magnitude_heatmap_base64

        t0 = time.perf_counter()
        embeddings = self.feature_extractor.extract_features_pil(images, batch_size=batch_size)
        probs = self._predict_embeddings(embeddings)
        total_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        avg_latency = round(total_time_ms / len(images), 2)

        results = []
        for i in range(len(images)):
            real_prob = float(probs[i, 0])
            ai_prob = float(probs[i, 1])
            label = "AI_GENERATED" if ai_prob >= 0.5 else "REAL"
            confidence = max(real_prob, ai_prob)

            res = {
                "label": label,
                "ai_probability": round(ai_prob, 4),
                "real_probability": round(real_prob, 4),
                "confidence": round(confidence, 4),
                "latency_ms": avg_latency,
            }

            if include_fft:
                res["fft_spectrum"] = [
                    round(float(v), 4)
                    for v in compute_radial_fft_spectrum(images[i], num_bins=32)
                ]
                res["fft_heatmap"] = generate_fft_magnitude_heatmap_base64(images[i], size=128)

            results.append(res)
        return results
