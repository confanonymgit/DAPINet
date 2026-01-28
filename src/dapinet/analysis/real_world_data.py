import glob
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dapinet.architecture.inference import ensemble_predict, load_models
from dapinet.clustering import ALGO_ORDER
from dapinet.utils import apply_standard_scaling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("compare_algo_selection")


def load_datasets(data_dir: Path) -> dict[str, dict]:
    """Load datasets from consolidated .npz files."""
    logger.info(f"Loading datasets from {data_dir}")

    npz_files = sorted(glob.glob(str(data_dir / "*.npz")))
    # Filter out non-dataset files
    npz_files = [f for f in npz_files if not Path(f).stem.startswith(("benchmark", "cvi"))]

    if not npz_files:
        logger.warning(f"No .npz dataset files found in {data_dir}")
        return {}

    data_items = {}
    for f in tqdm(npz_files, desc="Loading Datasets"):
        try:
            ds_name = Path(f).stem
            with np.load(f) as data:
                item = {
                    "X": data["X"].astype(np.float32),
                    "y": data["y"],
                }
                # Load ARI values if available
                if "ari" in data.keys():
                    item["ari"] = data["ari"]
                data_items[ds_name] = item
        except Exception as e:
            logger.error(f"Error loading {f}: {e}")

    logger.info(f"Loaded {len(data_items)} datasets.")
    return data_items


def run_model_inference(model_path: Path, datasets: dict) -> tuple[pd.DataFrame, dict]:
    """Run inference on datasets and return per-dataset scores."""

    models = load_models(str(model_path))
    if not models:
        logger.warning(f"No models loaded from {model_path}, skipping")

    rows: list[dict] = []
    inference_times: list[float] = []

    for ds_name, ds_data in tqdm(datasets.items(), desc="Model Inference"):
        X = ds_data["X"]
        X_normalized = apply_standard_scaling(X)

        start_time = time.perf_counter()
        scores = ensemble_predict(models, X_normalized)
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000
        inference_times.append(inference_time_ms)

        scores_arr = np.asarray(scores, dtype=float).reshape(-1)
        if scores_arr.size < len(ALGO_ORDER):
            logger.warning(
                f"Inference output for {ds_name} has {scores_arr.size} values; "
                f"expected {len(ALGO_ORDER)}"
            )

        row: dict = {
            "dataset": ds_name,
            "n_rows": int(X.shape[0]),
            "n_cols": int(X.shape[1]),
            "inference_time_ms": float(inference_time_ms),
        }

        # Predicted per-algorithm values
        for idx, algo in enumerate(ALGO_ORDER):
            if idx < scores_arr.size:
                row[f"{algo}"] = float(scores_arr[idx])
            else:
                row[f"{algo}"] = np.nan

        rows.append(row)

    timing_stats = {
        "total_time_ms": float(sum(inference_times)) if inference_times else 0.0,
        "mean_time_ms": float(np.mean(inference_times)) if inference_times else 0.0,
        "median_time_ms": float(np.median(inference_times)) if inference_times else 0.0,
        "min_time_ms": float(min(inference_times)) if inference_times else 0.0,
        "max_time_ms": float(max(inference_times)) if inference_times else 0.0,
        "n_datasets": int(len(inference_times)),
    }

    df = pd.DataFrame(rows)

    # Stable, readable column ordering
    base_cols = [
        "dataset",
        "inference_time_ms",
    ]
    pred_cols = [f"{algo}" for algo in ALGO_ORDER]

    ordered_cols = [c for c in base_cols if c in df.columns]
    ordered_cols += [c for c in pred_cols if c in df.columns]
    df = df[ordered_cols]

    return df, timing_stats
