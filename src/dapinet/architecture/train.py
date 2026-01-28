"""Core training and validation utilities."""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm

from .config import Config
from .dataset import create_dataloaders_from_samples
from .utils import save_checkpoint

logger = logging.getLogger(__name__)


def create_model() -> nn.Module:
    """Create the configured model."""
    if Config.MODEL_TYPE == "default":
        from .model import DoubleInvariantTransformer

        return DoubleInvariantTransformer()
    else:
        from .ablation_model import create_ablation_model

        return create_ablation_model(Config.MODEL_TYPE)


def validate(model, loader, criterion):
    """Evaluate on a validation loader."""

    model.eval()
    total_loss = 0
    steps = 0

    with torch.no_grad():
        for x, y, r_mask, c_mask in loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            r_mask, c_mask = r_mask.to(Config.DEVICE), c_mask.to(Config.DEVICE)

            logits = model(x, r_mask, c_mask)
            loss = criterion(logits, y)

            total_loss += loss.item()
            steps += 1

    return total_loss / steps


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    epoch_idx,
    fold_idx,
):
    """Train for one epoch and return avg loss."""
    model.train()
    total_loss = 0

    progress_bar = tqdm(loader, desc=f"Fold {fold_idx + 1} | Epoch {epoch_idx + 1}/{Config.EPOCHS}")

    for i, (x, y, r_mask, c_mask) in enumerate(progress_bar):
        x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
        r_mask, c_mask = r_mask.to(Config.DEVICE), c_mask.to(Config.DEVICE)

        logits = model(x, r_mask, c_mask)
        loss = criterion(logits, y)

        loss = loss / Config.ACCUM_STEPS
        loss.backward()

        if (i + 1) % Config.ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * Config.ACCUM_STEPS
        progress_bar.set_postfix(loss=loss.item() * Config.ACCUM_STEPS)

    return total_loss / len(loader)


def train_fold(
    train_loader,
    val_loader,
    fold_idx,
    output_dir,
):
    """Train one fold and return best loss and history."""

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = create_model().to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    counter = 0
    best_checkpoint_path = None

    for epoch in range(Config.EPOCHS):
        avg_train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch,
            fold_idx,
        )
        scheduler.step()

        avg_val_loss = validate(model, val_loader, criterion)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        logger.info(f"   Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = Path(checkpoint_dir) / f"checkpoint_fold_{fold_idx + 1}.pth"
            save_checkpoint(
                model, optimizer, epoch, avg_val_loss, best_checkpoint_path, fold_idx=fold_idx
            )
            counter = 0
        else:
            counter += 1
            if counter >= Config.PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    if best_checkpoint_path and best_checkpoint_path.exists():
        logger.info(f"Restoring best model from {best_checkpoint_path.name}")
        checkpoint = torch.load(
            best_checkpoint_path, map_location=Config.DEVICE, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

    return best_val_loss, history


def execute_kfold_training(
    train_samples: list[dict],
    output_dir: Path,
) -> tuple[list[float], list[dict]]:
    """Run K-fold training with file-level splits."""

    logger = logging.getLogger(__name__)

    unique_files = sorted(list(set(s["file_path"] for s in train_samples)))
    files_arr = np.array(unique_files)

    logger.info(
        f"K-Fold Cross Validation on {len(unique_files)} files ({len(train_samples)} samples)"
    )

    kf = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=Config.SEED)

    fold_results: list[float] = []
    all_histories: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(files_arr)):
        fold_start_time = time.time()

        logger.info(f"\nFOLD {fold + 1}/{Config.K_FOLDS}")

        fold_train_files = set(files_arr[train_idx])
        fold_val_files = set(files_arr[val_idx])

        fold_train_samples = [s for s in train_samples if s["file_path"] in fold_train_files]
        fold_val_samples = [s for s in train_samples if s["file_path"] in fold_val_files]

        train_loader, val_loader = create_dataloaders_from_samples(
            fold_train_samples,
            fold_val_samples,
        )

        best_val_loss, history = train_fold(
            train_loader,
            val_loader,
            fold,
            output_dir,
        )

        fold_results.append(best_val_loss)
        all_histories.append(history)

        # Log fold completion
        fold_duration = time.time() - fold_start_time
        logger.info(
            f"Fold {fold + 1} Best Val Loss: {best_val_loss:.6f} (Duration: {fold_duration:.2f}s)"
        )

    return fold_results, all_histories
