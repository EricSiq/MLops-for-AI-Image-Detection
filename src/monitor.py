"""
MLOps Monitoring Module: Embedding Drift and Prediction Distribution Drift Detection.
Utilizes Evidently AI and SciPy statistical tests to generate standalone HTML drift reports.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp, wasserstein_distance

from src.config import settings
from src.dataset import CIFAKEDatasetLoader
from src.model import AIImageClassifier
from src.utils import logger


def compute_distribution_drift(
    reference_data: np.ndarray,
    current_data: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes statistical drift metrics between reference (train) and current (production/scraped)
    distributions using Kolmogorov-Smirnov test and Wasserstein distance.
    """
    results = {}
    n_features = reference_data.shape[1]

    ks_pvalues = []
    wasserstein_dists = []

    for col in range(n_features):
        ref_col = reference_data[:, col]
        cur_col = current_data[:, col]

        # 2-sample KS test
        ks_res = ks_2samp(ref_col, cur_col)
        ks_pvalues.append(float(ks_res.pvalue))

        # Wasserstein (Earth Mover's) Distance
        w_dist = float(wasserstein_distance(ref_col, cur_col))
        wasserstein_dists.append(w_dist)

    # Drift is detected if p-value < 0.05 on a significant portion of features
    drifted_features = [i for i, p in enumerate(ks_pvalues) if p < 0.05]
    drift_share = len(drifted_features) / n_features

    results["drift_detected"] = bool(drift_share > 0.2)
    results["drift_share"] = round(float(drift_share), 4)
    results["mean_wasserstein_distance"] = round(float(np.mean(wasserstein_dists)), 4)
    results["max_wasserstein_distance"] = round(float(np.max(wasserstein_dists)), 4)
    results["tested_features_count"] = n_features
    results["drifted_features_count"] = len(drifted_features)

    return results


def generate_drift_report(
    reference_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
    reference_labels: Optional[np.ndarray] = None,
    current_predictions: Optional[np.ndarray] = None,
    output_html_path: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generates a comprehensive drift report comparing reference vs current data.
    Uses Evidently AI if available, and always outputs statistical metrics and standalone HTML.
    """
    html_path = output_html_path or settings.drift_report_html
    json_path = output_json_path or settings.drift_metrics_json
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Statistical drift computation
    # Use top 16 dimensions or PCA components for concise metrics
    n_dims = min(16, reference_embeddings.shape[1])
    drift_metrics = compute_distribution_drift(
        reference_embeddings[:, :n_dims],
        current_embeddings[:, :n_dims],
    )

    if reference_labels is not None and current_predictions is not None:
        drift_metrics["reference_ai_ratio"] = round(float(np.mean(reference_labels)), 4)
        drift_metrics["current_ai_ratio"] = round(float(np.mean(current_predictions)), 4)

    # 2. Try generating Evidently report
    evidently_success = False
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        ref_df = pd.DataFrame(
            reference_embeddings[:, :n_dims],
            columns=[f"embed_{i}" for i in range(n_dims)],
        )
        cur_df = pd.DataFrame(
            current_embeddings[:, :n_dims],
            columns=[f"embed_{i}" for i in range(n_dims)],
        )

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        report.save_html(str(html_path))
        evidently_success = True
        logger.info(f"Evidently AI HTML drift report generated at {html_path}.")
    except Exception as err:
        logger.warning(f"Could not generate report via Evidently ({err}). Generating custom HTML report...")

    # 3. Fallback / Custom HTML report if Evidently was not used or failed
    if not evidently_success:
        html_report = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>MLOps Embedding & Prediction Drift Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
    .container {{ max-width: 900px; margin: 0 auto; background: #1e293b; padding: 2rem; border-radius: 12px; border: 1px solid #334155; }}
    h1 {{ color: #38bdf8; margin-bottom: 0.5rem; }}
    .status {{ display: inline-block; padding: 0.4rem 1rem; border-radius: 9999px; font-weight: bold; margin-bottom: 1.5rem; }}
    .status.ok {{ background: #065f46; color: #6ee7b7; }}
    .status.drift {{ background: #991b1b; color: #fca5a5; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #334155; }}
    th {{ background: #0f172a; color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Data & Embedding Drift Report</h1>
    <p style="color: #94a3b8; margin-bottom: 1.5rem;">Comparing Reference Training Set vs Live Scraped Batch</p>
    <div class="status {'drift' if drift_metrics['drift_detected'] else 'ok'}">
      {'DRIFT DETECTED' if drift_metrics['drift_detected'] else 'STABLE (NO CRITICAL DRIFT)'}
    </div>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Drift Detected</td><td>{drift_metrics['drift_detected']}</td></tr>
      <tr><td>Drifted Features Share</td><td>{drift_metrics['drift_share'] * 100:.1f}%</td></tr>
      <tr><td>Mean Wasserstein Distance</td><td>{drift_metrics['mean_wasserstein_distance']}</td></tr>
      <tr><td>Max Wasserstein Distance</td><td>{drift_metrics['max_wasserstein_distance']}</td></tr>
      <tr><td>Tested Features</td><td>{drift_metrics['tested_features_count']}</td></tr>
      <tr><td>Reference AI Ratio</td><td>{drift_metrics.get('reference_ai_ratio', 'N/A')}</td></tr>
      <tr><td>Current Live AI Ratio</td><td>{drift_metrics.get('current_ai_ratio', 'N/A')}</td></tr>
    </table>
  </div>
</body>
</html>
        """
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_report)
        logger.info(f"Custom HTML drift report generated at {html_path}.")

    # Write metrics JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(drift_metrics, f, indent=2)

    logger.info(f"Drift metrics saved to {json_path}.")
    return drift_metrics
