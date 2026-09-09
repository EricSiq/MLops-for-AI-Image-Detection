"""
Training pipeline with MLflow experiment tracking, evaluation, and model registry tagging.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import settings
from src.dataset import CIFAKEDatasetLoader
from src.model import AIImageClassifier, CLIPFeatureExtractor
from src.utils import format_metrics, logger


def evaluate_model(
    classifier: AIImageClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> Tuple[Dict[str, float], Path, Path]:
    """
    Computes comprehensive evaluation metrics and saves confusion matrix and ROC curve plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "AI"],
        yticklabels=["Real", "AI"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    return metrics, cm_path, roc_path


def run_training(
    sample_size: int = 5000,
    batch_size: int = 64,
    C: float = 1.0,
    max_iter: int = 1000,
    export_onnx: bool = True,
) -> Dict[str, float]:
    """
    Full end-to-end training and tracking pipeline.
    """
    import mlflow
    import mlflow.sklearn

    # Configure MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    logger.info(f"Starting MLflow Run (experiment='{settings.mlflow_experiment_name}')...")

    # 1. Dataset & Embedding Extraction
    dataset_loader = CIFAKEDatasetLoader()
    feature_extractor = CLIPFeatureExtractor()

    # Train embeddings
    cached_train = dataset_loader.load_cached_embeddings("train", sample_size)
    if cached_train is not None:
        X_train, y_train = cached_train
    else:
        logger.info(f"Loading raw training images (sample_size={sample_size})...")
        train_images, y_train = dataset_loader.load_dataset(sample_size=sample_size, split="train")
        X_train = feature_extractor.extract_features_pil(train_images, batch_size=batch_size)
        dataset_loader.save_cached_embeddings("train", sample_size, X_train, y_train)

    # Test embeddings
    test_sample_size = max(500, sample_size // 4)
    cached_test = dataset_loader.load_cached_embeddings("test", test_sample_size)
    if cached_test is not None:
        X_test, y_test = cached_test
    else:
        logger.info(f"Loading raw test images (sample_size={test_sample_size})...")
        test_images, y_test = dataset_loader.load_dataset(
            sample_size=test_sample_size, split="test"
        )
        X_test = feature_extractor.extract_features_pil(test_images, batch_size=batch_size)
        dataset_loader.save_cached_embeddings("test", test_sample_size, X_test, y_test)

    # 2. Train Classifier Head
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(
            {
                "backbone": settings.clip_model_name,
                "embedding_dim": X_train.shape[1],
                "sample_size_train": X_train.shape[0],
                "sample_size_test": X_test.shape[0],
                "C": C,
                "max_iter": max_iter,
                "solver": "lbfgs",
                "random_seed": settings.random_seed,
            }
        )

        classifier = AIImageClassifier(C=C, max_iter=max_iter)
        classifier.fit(X_train, y_train)

        # 3. Evaluate
        metrics, cm_path, roc_path = evaluate_model(
            classifier, X_test, y_test, output_dir=settings.temp_dir
        )
        logger.info(format_metrics(metrics))

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Log artifact figures
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(roc_path), artifact_path="evaluation_plots")

        # Save local joblib model with checksum
        model_save_path = classifier.save(settings.joblib_model_path)

        # Infer model signature for schema enforcement
        from mlflow.models.signature import infer_signature

        signature = infer_signature(
            X_train[:5], classifier.model.predict_proba(X_train[:5])
        )

        # Log model to MLflow model registry
        mlflow.sklearn.log_model(
            sk_model=classifier.model,
            artifact_path="classifier_model",
            registered_model_name=settings.mlflow_model_name,
            signature=signature,
            input_example=X_train[:2],
        )

        logger.info(f"MLflow Run ID: {run.info.run_id} completed successfully.")

        # 4. Optional ONNX export
        if export_onnx:
            try:
                from src.inference import export_classifier_to_onnx

                export_classifier_to_onnx(classifier, settings.onnx_model_path)
                mlflow.log_artifact(str(settings.onnx_model_path), artifact_path="onnx_model")
            except Exception as e:
                logger.warning(f"Could not export ONNX model: {e}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Image Detector Classifier")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=settings.default_sample_size,
        help="Number of samples to train on (-1 for full dataset)",
    )
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--no-onnx", action="store_true", help="Disable ONNX export")

    args = parser.parse_args()
    run_training(
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        C=args.C,
        max_iter=args.max_iter,
        export_onnx=not args.no_onnx,
    )
