import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def cached_load_npz(path: str) -> dict | None:
    """Load and cache an NPZ file."""
    try:
        with np.load(path, allow_pickle=True) as data:
            return dict(data)
    except Exception as e:
        logger.error("Error loading NPZ file from %s: %s", path, e)
        return None


def load_samples_from_parquet(parquet_path: Path | str, split: str = "train") -> list[dict]:
    """Load sample rows from a parquet index for a split."""

    df = pd.read_parquet(parquet_path)

    logger.info(f"Loaded dataset index with {len(df)} samples from {parquet_path}")

    df_filtered = df[df["split"] == split]

    logger.info(f"Loaded {len(df_filtered)} samples for split '{split}'")

    return df_filtered.to_dict("records")


class DynamicNpzDataset(Dataset):
    """Dataset backed by indexed NPZ samples."""

    def __init__(self, samples: list[dict]):
        """Create dataset from sample dicts."""
        self.samples = samples
        logger.info(f"Dataset initialized with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path = sample["file_path"]
        key_x = sample["sample_key"]
        key_y = sample["target_key"]

        npz_content = cached_load_npz(file_path)
        if npz_content is None:
            logger.warning(f"Failed to load {file_path}, returning zeros")
            return (
                torch.zeros((Config.MAX_ROWS, Config.MAX_COLS)),
                torch.zeros(Config.NUM_CLASSES),
            )

        try:
            data = npz_content[key_x].astype(np.float32)
            target = np.clip(npz_content[key_y].astype(np.float32), 0.0, 1.0)
        except KeyError:
            logger.error(f"Key missing in {file_path}: {key_x} or {key_y}")
            return (
                torch.zeros((Config.MAX_ROWS, Config.MAX_COLS)),
                torch.zeros(Config.NUM_CLASSES),
            )

        return torch.FloatTensor(data), torch.FloatTensor(target)


def collate_batch(batch):
    """Pad a batch to its max rows/cols."""
    max_rows = max(item[0].shape[0] for item in batch)
    max_cols = max(item[0].shape[1] for item in batch)

    max_rows = min(max_rows, Config.MAX_ROWS)
    max_cols = min(max_cols, Config.MAX_COLS)

    data_list = []
    target_list = []
    row_mask_list = []
    col_mask_list = []

    for data, target in batch:
        rows, cols = data.shape

        padded_data = torch.zeros((max_rows, max_cols), dtype=torch.float32)

        r_mask = torch.ones(max_rows, dtype=torch.bool)
        c_mask = torch.ones(max_cols, dtype=torch.bool)

        r = min(rows, max_rows)
        c = min(cols, max_cols)

        padded_data[:r, :c] = data[:r, :c]

        r_mask[:r] = False
        c_mask[:c] = False

        data_list.append(padded_data)
        target_list.append(target)
        row_mask_list.append(r_mask)
        col_mask_list.append(c_mask)

    return (
        torch.stack(data_list),
        torch.stack(target_list),
        torch.stack(row_mask_list),
        torch.stack(col_mask_list),
    )


def create_dataloaders_from_samples(
    train_samples: list[dict],
    val_samples: list[dict],
    shuffle_train: bool = True,
) -> tuple:
    """Create train/val DataLoaders from sample lists."""

    logger = logging.getLogger(__name__)

    logger.info(f"  Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    train_dataset = DynamicNpzDataset(train_samples)
    val_dataset = DynamicNpzDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle_train,
        collate_fn=collate_batch,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False,
        prefetch_factor=4 if Config.NUM_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False,
        prefetch_factor=4 if Config.NUM_WORKERS > 0 else None,
    )

    return train_loader, val_loader
