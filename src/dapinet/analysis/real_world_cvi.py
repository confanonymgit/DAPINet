import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from dapinet.clustering import ALGO_ORDER
from dapinet.utils import apply_standard_scaling, dunn_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("cvi_evaluation")


# CVI definitions: name -> (function, higher_is_better)
CVI_METRICS = {
    "silhouette": (silhouette_score, True),  # higher is better
    "calinski_harabasz": (calinski_harabasz_score, True),  # higher is better
    "davies_bouldin": (davies_bouldin_score, False),  # lower is better
    "dunn": (dunn_score, True),  # higher is better
}


def _calculate_cvi(X: np.ndarray, labels: np.ndarray, metric_name: str) -> float | None:
    """Calculate a single CVI metric, returning None on failure."""
    metric_func, _ = CVI_METRICS[metric_name]

    # Check if we have valid labels (at least 2 clusters, not all same)
    unique_labels = np.unique(labels)
    # Filter out noise labels (-1)
    valid_labels = unique_labels[unique_labels >= 0]

    if len(valid_labels) < 2:
        return None

    # For metrics that don't handle noise labels well, mask them
    mask = labels >= 0
    if mask.sum() < 2:
        return None

    X_valid = X[mask]
    labels_valid = labels[mask]

    # Check again after masking
    if len(np.unique(labels_valid)) < 2:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = metric_func(X_valid, labels_valid)
        return float(score)
    except Exception:
        return None


def evaluate_dataset(npz_path: Path, scale: bool = True) -> dict:
    """Evaluate all algorithms on a single dataset."""
    with np.load(npz_path) as data:
        X = data["X"]

        # Get all algorithm predictions
        algo_preds = {}
        for algo in ALGO_ORDER:
            if algo in data.keys():
                algo_preds[algo] = data[algo]

    ds_name = npz_path.stem

    # Optionally scale features
    if scale:
        X = apply_standard_scaling(X)

    results = {"dataset": ds_name}

    for cvi_name in CVI_METRICS.keys():
        start_time = time.perf_counter()
        for algo_name, labels in algo_preds.items():
            key = f"{algo_name}_{cvi_name}"
            results[key] = _calculate_cvi(X, labels, cvi_name)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        results[f"{cvi_name}_time_ms"] = elapsed_ms

    return results


def find_best_algorithms(df: pd.DataFrame) -> pd.DataFrame:
    """Find the best algorithm for each dataset based on each CVI."""
    best_results = []

    for _, row in df.iterrows():
        ds_name = row["dataset"]
        best_row = {"dataset": ds_name}

        for cvi_name, (_, higher_is_better) in CVI_METRICS.items():
            # Get all values for this CVI
            cvi_values = {}
            for algo in ALGO_ORDER:
                key = f"{algo}_{cvi_name}"
                if key in row and pd.notna(row[key]):
                    cvi_values[algo] = row[key]

            if cvi_values:
                if higher_is_better:
                    best_algo = max(cvi_values.items(), key=lambda x: x[1])
                else:
                    best_algo = min(cvi_values.items(), key=lambda x: x[1])

                best_row[f"{cvi_name}_algorithm"] = best_algo[0]
                best_row[f"{cvi_name}_value"] = best_algo[1]
            else:
                best_row[f"{cvi_name}_algorithm"] = None
                best_row[f"{cvi_name}_value"] = None

        best_results.append(best_row)

    return pd.DataFrame(best_results)


def create_per_cvi_matrix(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create separate matrices for each CVI (datasets × algorithms)."""
    matrices = {}

    for cvi_name in CVI_METRICS.keys():
        data = []
        for _, row in df.iterrows():
            row_data = {"dataset": row["dataset"]}
            for algo in ALGO_ORDER:
                key = f"{algo}_{cvi_name}"
                row_data[algo] = row.get(key, None)
            data.append(row_data)

        matrices[cvi_name] = pd.DataFrame(data)

    return matrices
