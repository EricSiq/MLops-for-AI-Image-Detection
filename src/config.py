"""
Configuration module for the AI Image Detection MLOps Pipeline.
Uses Pydantic BaseSettings for centralized, type-safe settings with environment variable overrides.
"""

from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MLOPS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    mlruns_dir: Path = PROJECT_ROOT / "mlruns"
    reports_dir: Path = PROJECT_ROOT / "reports"
    temp_dir: Path = PROJECT_ROOT / "temp"

    # Dataset Settings
    dataset_name: str = "batgre/CIFAKE"
    dataset_revision: str = "main"
    random_seed: int = 42
    default_sample_size: int = 5000  # -1 for full dataset
    test_split_ratio: float = 0.2

    # Model & Feature Extractor
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_revision: str = "main"  # Pinned commit SHA or tag for supply chain safety
    device: str = "cuda"  # Auto-falls back to cpu if cuda not available
    batch_size: int = 64
    embedding_dim: int = 512
    onnx_model_path: Path = PROJECT_ROOT / "models" / "classifier_head.onnx"
    joblib_model_path: Path = PROJECT_ROOT / "models" / "classifier_head.joblib"

    # MLflow Settings (Uses SQLite backend for MLflow 3.x model registry support)
    mlflow_tracking_uri: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow_experiment_name: str = "ai-image-detector"
    mlflow_model_name: str = "ai-image-detector-clip"

    # Scraper & Ingestion Settings
    max_images_per_scrape: int = 50
    scrape_timeout_seconds: int = 10
    max_redirects: int = 5
    min_image_width: int = 32
    min_image_height: int = 32
    max_image_bytes: int = 15 * 1024 * 1024  # 15MB limit per image
    allowed_image_mimes: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/avif",
    ]
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 AIImageDetector/1.0"
    )
    # SSRF Protection: Denied IP ranges and private hostnames
    denied_hosts: List[str] = Field(
        default=[
            "localhost",
            "127.0.0.1",
            "::1",
            "0.0.0.0",  # nosec B104 - Blacklist entry for SSRF filter, not a bind address
            "metadata.google.internal",
            "169.254.169.254",
        ]
    )

    # Serving / API Settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_title: str = "AI-Generated Image Detector API"
    api_version: str = "0.1.0"

    # Monitoring Settings
    drift_pvalue_threshold: float = 0.05
    drift_share_threshold: float = 0.20
    drift_report_html: Path = PROJECT_ROOT / "reports" / "drift_report.html"
    drift_metrics_json: Path = PROJECT_ROOT / "reports" / "drift_metrics.json"

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for path in [
            self.data_dir,
            self.models_dir,
            self.mlruns_dir,
            self.reports_dir,
            self.temp_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
