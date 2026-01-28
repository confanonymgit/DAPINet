import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dapinet.architecture.inference import ensemble_predict, load_models


def load_test_datasets_from_index(parquet_path: Path) -> pd.DataFrame:
    """Load test split from parquet index."""
    df = pd.read_parquet(parquet_path)

    # Remove Linux path prefix
    linux_prefix = "/lustre06/project/6006810/mrb/projects/dapinet/"
    df["file_path"] = df["file_path"].str.replace(linux_prefix, "", regex=False)

    # Filter test split
    test_df = df[df["split"] == "test"].copy()

    print(f"Total datasets in index: {len(df)}")
    print(f"Test datasets: {len(test_df)}")

    return test_df


def run_inference_on_test_datasets(
    model_dir: Path, test_df: pd.DataFrame, verbose: bool = True
) -> pd.DataFrame:
    """Run inference on test datasets and return results."""

    models = load_models(model_dir)
    if not models:
        raise FileNotFoundError(
            f"No model checkpoints (.pth) found in: {model_dir}. "
            "Set model_dir to a directory containing checkpoint_fold_*.pth files."
        )
    results = []

    # Track loaded files to avoid reloading
    loaded_files = {}
    loaded_configs = {}

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing datasets"):
        file_path = row["file_path"]
        sample_key = row["sample_key"]
        target_key = row["target_key"]

        try:
            # Load file (cache to avoid reloading)
            if file_path not in loaded_files:
                if not Path(file_path).exists():
                    print(f"Warning: File not found: {file_path}")
                    continue
                loaded_files[file_path] = np.load(file_path)

            # Load config to get num_clusters
            config_path = Path(file_path).parent / (
                Path(file_path).stem.replace("_training", "") + ".json"
            )
            if config_path not in loaded_configs:
                try:
                    if config_path.exists():
                        with open(config_path) as f:
                            loaded_configs[config_path] = json.load(f)
                    else:
                        loaded_configs[config_path] = {}
                except Exception:
                    loaded_configs[config_path] = {}

            config = loaded_configs.get(config_path, {})
            num_clusters = config.get("cluster_config", {}).get("num_clusters", np.nan)

            data = loaded_files[file_path]

            # Extract sample (input) and target (ground truth)
            X = data[sample_key].astype(np.float32)
            y_true = data[target_key]

            # Run model inference
            start_time = time.perf_counter()
            probs = ensemble_predict(models, X)
            end_time = time.perf_counter()

            inference_time_ms = (end_time - start_time) * 1000

            # Calculate MSE between predicted probabilities and ground truth
            mse = np.mean((probs - y_true) ** 2)

            # Store results
            results.append(
                {
                    "file_path": file_path,
                    "sample_key": sample_key,
                    "target_key": target_key,
                    "ARI": y_true,
                    "predicted_probs": probs,
                    "mse_loss": mse,
                    "inference_time_ms": inference_time_ms,
                    "num_samples": X.shape[0],
                    "num_features": X.shape[1],
                    "num_clusters": num_clusters,
                }
            )

        except Exception as e:
            print(f"Error processing {file_path}/{sample_key}: {e}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    return results_df
