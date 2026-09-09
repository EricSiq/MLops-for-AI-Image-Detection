"""
Utility functions for logging, metrics, hashing, and safe file I/O.
"""

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict


def get_logger(name: str) -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


logger = get_logger("mlops-pipeline")


def compute_sha256(data: bytes) -> str:
    """Calculates SHA256 checksum for byte content."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Formats a dictionary of evaluation metrics for clean display."""
    lines = ["=" * 40, "Evaluation Metrics Summary", "=" * 40]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key:<20}: {value:.4f}")
        else:
            lines.append(f"  {key:<20}: {value}")
    lines.append("=" * 40)
    return "\n".join(lines)
