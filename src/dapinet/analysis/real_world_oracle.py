import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dapinet.clustering import ALGO_ORDER, run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("benchmark")


def load_dataset(npz_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Load dataset from consolidated .npz file."""
    with np.load(npz_path) as data:
        X = data["X"]
        y = data["y"]
    ds_name = npz_path.stem
    return X, y, ds_name


def save_dataset(
    npz_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    ari: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    """Save dataset with results to consolidated .npz file."""
    file_data = {"X": X, "y": y, "ari": ari, **predictions}
    np.savez_compressed(npz_path, **file_data)


def run_benchmarks(
    data_dir: str,
    n_trials: int = 50,
    seed: int = 42,
    n_jobs: int = 1,
    timeout: int = 60,
) -> None:
    """Run clustering benchmarks on all datasets in directory."""
    data_path = Path(data_dir)

    # Find consolidated .npz files
    dataset_files = sorted(glob.glob(str(data_path / "*.npz")))

    # Filter out non-dataset files
    dataset_files = [f for f in dataset_files if not Path(f).stem.startswith("benchmark")]

    if not dataset_files:
        logger.error(f"No .npz dataset files found in {data_dir}")
        return

    # Output paths
    output_json = Path("results/oracle.json")
    output_csv = Path("results/oracle_ari.csv")
    output_time_csv = Path("results/oracle_time.csv")

    all_results = {}
    ari_rows = []
    time_rows = []

    logger.info(
        f"Running benchmark on {len(dataset_files)} datasets with {len(ALGO_ORDER)} algorithms each"
    )
    logger.info(f"Settings: n_trials={n_trials}, timeout={timeout}s, seed={seed}")

    for ds_file in tqdm(dataset_files, desc="Processing Datasets"):
        ds_path = Path(ds_file)

        try:
            # Load dataset
            X, y, ds_name = load_dataset(ds_path)

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {ds_name}...")
            logger.info(
                f"Shape: {X.shape[0]} rows x {X.shape[1]} cols, {len(np.unique(y))} clusters"
            )
            logger.info(f"{'=' * 60}")

            # Preprocessing: using centralized StandardScaler utility
            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X)

            # Run experiment for ALL algorithms
            results = run_experiment(
                X=X_norm,
                y=y,
                n_trials_per_algo=n_trials,
                seed=seed,
                algorithms=ALGO_ORDER,
                optuna_n_jobs=n_jobs,
                timeout=timeout,
            )

            # Collect results
            ds_result_dict = {}
            ari_row = {"dataset": ds_name}
            time_row = {"dataset": ds_name}
            predictions = {}
            ari_vector = []

            for algo_name in ALGO_ORDER:
                if algo_name in results:
                    res = results[algo_name]
                    ds_result_dict[algo_name] = {
                        "best_ari": float(res.best_ari),
                        "best_params": res.best_params,
                        "duration": res.duration,
                    }
                    ari_row[algo_name] = float(res.best_ari)
                    ari_vector.append(float(res.best_ari))
                    time_row[algo_name] = float(res.duration)

                    if res.prediction is not None:
                        predictions[algo_name] = res.prediction
                else:
                    ds_result_dict[algo_name] = {
                        "best_ari": None,
                        "best_params": None,
                        "duration": None,
                    }
                    ari_row[algo_name] = None
                    ari_vector.append(-1.0)
                    time_row[algo_name] = None

            # Save consolidated dataset file
            save_dataset(
                ds_path,
                X=X,
                y=y,
                ari=np.array(ari_vector, dtype=np.float64),
                predictions=predictions,
            )
            logger.info(f"Updated {ds_path.name} with {len(predictions)} predictions")

            all_results[ds_name] = ds_result_dict
            ari_rows.append(ari_row)
            time_rows.append(time_row)

            # Find best algorithm
            best_algo = max(
                [
                    (algo, ds_result_dict[algo]["best_ari"])
                    for algo in ALGO_ORDER
                    if ds_result_dict[algo]["best_ari"] is not None
                ],
                key=lambda x: x[1],
                default=(None, -1),
            )
            logger.info(f"Finished {ds_name}. Best: {best_algo[0]} (ARI={best_algo[1]:.4f})")

            # Save intermediate results
            with open(output_json, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            df_ari = pd.DataFrame(ari_rows)
            df_time = pd.DataFrame(time_rows)
            cols_order = ["dataset"] + list(ALGO_ORDER)

            df_ari = df_ari[[c for c in cols_order if c in df_ari.columns]]
            df_ari.to_csv(output_csv, index=False)

            df_time = df_time[[c for c in cols_order if c in df_time.columns]]
            df_time.to_csv(output_time_csv, index=False)

        except Exception as e:
            logger.error(f"Failed to process {ds_path.name}: {e}")
            import traceback

            traceback.print_exc()

    # Final save
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    df_ari = pd.DataFrame(ari_rows)
    df_time = pd.DataFrame(time_rows)
    cols_order = ["dataset"] + list(ALGO_ORDER)

    df_ari = df_ari[[c for c in cols_order if c in df_ari.columns]]
    df_ari.to_csv(output_csv, index=False)

    df_time = df_time[[c for c in cols_order if c in df_time.columns]]
    df_time.to_csv(output_time_csv, index=False)

    logger.info(f"\n{'=' * 60}")
    logger.info("Benchmark complete!")
    logger.info(f"Results JSON: {output_json}")
    logger.info(f"ARI matrix CSV: {output_csv}")
    logger.info(f"Time matrix CSV: {output_time_csv}")
    logger.info(f"{'=' * 60}")

    # Print summary
    print("\n=== ARI Matrix ===")
    print(df_ari.to_string(index=False))

    if not df_ari.empty:
        algo_cols = [c for c in df_ari.columns if c != "dataset"]
        winners = df_ari[algo_cols].idxmax(axis=1)
        print("\n=== Best Algorithm Distribution ===")
        print(winners.value_counts().to_string())
