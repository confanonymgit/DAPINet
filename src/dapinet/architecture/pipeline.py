import logging
import time
from pathlib import Path

from .config import Config
from .dataset import load_samples_from_parquet
from .model import DoubleInvariantTransformer
from .train import execute_kfold_training
from .utils import (
    log_device_info,
    log_fold_results,
    log_model_summary,
    log_timing,
    log_training_configuration,
    plot_training_history,
    save_training_history,
    set_seed,
)

logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_root: Path | str | None = None,
    output_dir: str | Path = ".",
) -> tuple[list[float], list[dict]]:
    """Run the full training pipeline."""
    start_time = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(Config.SEED)

    logger.info("STARTING TRAINING")

    log_device_info()

    log_model_summary(DoubleInvariantTransformer)

    log_training_configuration()

    parquet_path = data_root / "dataset_index.parquet"

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Required dataset index not found at: {parquet_path}")

    logger.info("\nLoading data from parquet...")
    train_samples = load_samples_from_parquet(parquet_path, split="train")

    if not train_samples:
        raise ValueError("No training samples found.")

    logger.info(f"Total train samples: {len(train_samples)}")

    logger.info("\nSTARTING K-FOLD TRAINING LOOP")
    training_start_time = time.time()

    fold_results, all_histories = execute_kfold_training(
        train_samples=train_samples,
        output_dir=output_dir,
    )

    logger.info("\nTRAINING COMPLETE")

    training_duration = time.time() - training_start_time
    log_timing("Training Loop Duration", training_duration)

    log_fold_results(fold_results)

    save_training_history(all_histories, str(output_dir / "training_history.json"))
    plot_training_history(all_histories, str(output_dir))

    total_time = time.time() - start_time
    log_timing("Total Execution Time", total_time)

    logger.info(f"\nResults saved to: {output_dir.absolute()}")

    return fold_results, all_histories
