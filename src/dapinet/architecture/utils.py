import datetime
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchinfo import summary
from tqdm import tqdm

from .config import Config

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info(f"Random Seed set to: {seed}")


def save_checkpoint(
    model, optimizer, epoch, loss, path, fold_idx=None, config=None, additional_info=None
):
    """Save a checkpoint with metadata."""

    logger.info(f"Saving model to {path}...")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "config": Config.to_dict() if config is None else config,
        "fold_idx": fold_idx,
    }

    if additional_info:
        checkpoint["additional_info"] = additional_info

    torch.save(checkpoint, path)
    logger.info("✓ Checkpoint saved successfully")


def load_checkpoint(path, model, optimizer=None):
    logger.info(f"Loading model from {path}...")
    checkpoint = torch.load(path, map_location=Config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


def plot_training_history(all_histories: list, save_dir: str = ".") -> None:
    """Plot train/val loss curves per fold."""

    plt.figure(figsize=(12, 8))

    for fold_idx, history in enumerate(all_histories):
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.plot(
            epochs,
            history["train_loss"],
            label=f"Fold {fold_idx + 1} Train",
            linestyle="--",
        )
        plt.plot(epochs, history["val_loss"], label=f"Fold {fold_idx + 1} Val")

    plt.title("Training and Validation Loss per Fold")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    save_path = Path(save_dir) / "loss_curves.png"
    plt.savefig(save_path)
    logger.info(f"Loss curves saved to {save_path}")
    plt.close()


def save_training_history(all_histories: list, save_path: str = "training_history.json") -> None:
    """Save training histories to JSON."""
    with open(save_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    logger.info(f"Training history saved to {save_path}")


def log_device_info() -> None:
    """Logs device information (CPU/GPU)."""
    logger.info(f"Device: {Config.DEVICE}")
    if Config.DEVICE.type == "cuda":
        logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")


def log_training_configuration() -> None:
    """Logs training configuration."""
    logger.info("\nTraining Configuration:")
    logger.info(f"  D_MODEL: {Config.D_MODEL}")
    logger.info(f"  N_HEAD: {Config.N_HEAD}")
    logger.info(f"  N_LAYERS: {Config.N_LAYERS}")
    logger.info(f"  DROPOUT: {Config.DROPOUT}")
    logger.info(f"  LR: {Config.LEARNING_RATE}")
    logger.info(f"  WEIGHT_DECAY: {Config.WEIGHT_DECAY}")
    logger.info(f"  K-Folds: {Config.K_FOLDS}")
    logger.info(f"  Batch Size: {Config.BATCH_SIZE}")
    logger.info(f"  Accum Steps: {Config.ACCUM_STEPS}")
    logger.info(f"  Epochs: {Config.EPOCHS}")
    logger.info(f"  Patience: {Config.PATIENCE}")
    logger.info(f"  Num Workers: {Config.NUM_WORKERS}")
    logger.info(f"  Device: {Config.DEVICE}")


def log_model_summary(model_class, input_shapes: dict = None) -> None:
    """Log a torchinfo summary for a model class."""

    logger.info("--- Model Architecture & Data Flow ---")
    model = model_class()

    if input_shapes is None:
        input_shapes = {
            "x": (1, Config.MAX_ROWS, Config.MAX_COLS),
            "r_mask": (1, Config.MAX_ROWS),
            "c_mask": (1, Config.MAX_COLS),
        }

    dummy_x = torch.zeros(input_shapes["x"], dtype=torch.float32)
    dummy_r_mask = torch.zeros(input_shapes["r_mask"], dtype=torch.bool)
    dummy_c_mask = torch.zeros(input_shapes["c_mask"], dtype=torch.bool)

    results = summary(
        model,
        input_data=[dummy_x, dummy_r_mask, dummy_c_mask],
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        device="cpu",
        verbose=0,
    )

    summary_str = str(results)
    clean_summary = summary_str.encode("ascii", "replace").decode("ascii")
    logger.info("\n" + clean_summary)


def prepare_dataset_index(data_dir: Path, test_ratio: float = 0.2, seed: int = 42) -> None:
    """Create a file-level train/test index for NPZ samples."""

    logger.info(f"Creating dataset index for {data_dir}")

    files = list(data_dir.rglob("*_training.npz"))

    if not files:
        logger.warning(f"No NPZ files found in {data_dir}")
        return

    logger.info(f"Found {len(files)} NPZ files. splitting...")

    random.seed(seed)
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)

    n_test = int(len(files_shuffled) * test_ratio)
    n_train = len(files_shuffled) - n_test

    file_split_map = {}
    for i, file_path in enumerate(files_shuffled):
        split = "train" if i < n_train else "test"
        file_split_map[file_path] = split

    logger.info(f"Files split: {n_train} train, {n_test} test")
    logger.info("Scanning files for samples...")

    records = []

    for file_path, split in tqdm(file_split_map.items(), desc="Scanning files"):
        try:
            with np.load(file_path) as data:
                keys = set(data.keys())

            # Identify X keys and match with y keys
            x_keys = [k for k in keys if k.endswith("_X")]

            for key_x in x_keys:
                key_y = key_x.replace("_X", "_y")
                if key_y in keys:
                    records.append(
                        {
                            "file_path": str(file_path.absolute()),
                            "sample_key": key_x,
                            "target_key": key_y,
                            "split": split,
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to scan {file_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save to parquet
    index_file = data_dir / "dataset_index.parquet"
    df.to_parquet(index_file, index=False)

    logger.info(f"\n{'=' * 60}")
    logger.info("Created dataset index")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total samples: {len(df)}")

    if len(df) > 0:
        n_train_samples = (df["split"] == "train").sum()
        n_test_samples = (df["split"] == "test").sum()

        logger.info(f"Train samples: {n_train_samples} ({n_train_samples / len(df) * 100:.1f}%)")
        logger.info(f"Test samples: {n_test_samples} ({n_test_samples / len(df) * 100:.1f}%)")

    logger.info(f"\nSaved to: {index_file}")


def log_fold_results(fold_results: list[float]) -> None:
    """Log K-Fold cross-validation results summary."""
    logger.info("--- K-Fold Results ---")
    for i, loss in enumerate(fold_results):
        logger.info(f"Fold {i + 1}: {loss:.6f}")

    if fold_results:
        avg_loss = sum(fold_results) / len(fold_results)
        logger.info(f"Average Best Val Loss: {avg_loss:.6f}")


def log_timing(label: str, duration: float) -> None:
    """Log timing information in seconds and minutes."""
    logger.info(f"{label}: {duration:.2f}s ({duration / 60:.2f}m)")
