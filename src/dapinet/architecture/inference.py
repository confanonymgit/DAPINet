import glob
import logging
import os

import torch

from .ablation_model import create_ablation_model
from .config import Config
from .model import DoubleInvariantTransformer

logger = logging.getLogger(__name__)


def load_models(model_dir):
    """Load .pth checkpoints and return ready models."""
    models = []
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not model_files:
        logger.error(f"No model files found in {model_dir}")
        return []

    logger.info(f"Found {len(model_files)} models in {model_dir}")

    for model_path in model_files:
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=False)
            else:
                checkpoint = torch.load(
                    model_path, map_location=torch.device("cpu"), weights_only=False
                )

            if "config" in checkpoint:
                conf = checkpoint["config"]
                logger.info(f"Updating Config from {os.path.basename(model_path)}...")
                Config.from_dict(conf)

            if Config.MODEL_TYPE == "default":
                model = DoubleInvariantTransformer().to(Config.DEVICE)
            else:
                model = create_ablation_model(Config.MODEL_TYPE).to(Config.DEVICE)

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)

            epoch = checkpoint.get("epoch", "Unknown")
            loss = checkpoint.get("loss", 0.0)
            if isinstance(loss, (int, float)):
                loss_str = f"{loss:.4f}"
            else:
                loss_str = str(loss)

            logger.info(f"Loaded model from {model_path} (Epoch {epoch}, Loss {loss_str})")

        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")

    return models


def preprocess_table(table_data):
    """Prepare a table tensor and masks for inference."""

    rows, cols = table_data.shape

    r = min(rows, Config.MAX_ROWS)
    c = min(cols, Config.MAX_COLS)

    tensor_data = torch.from_numpy(table_data).float()

    r_mask = torch.zeros(r, dtype=torch.bool)
    c_mask = torch.zeros(c, dtype=torch.bool)

    return tensor_data.unsqueeze(0), r_mask.unsqueeze(0), c_mask.unsqueeze(0)


def ensemble_predict(models, table_data):
    """Average predictions from all models for one table."""
    x, r_mask, c_mask = preprocess_table(table_data)
    x = x.to(Config.DEVICE)
    r_mask = r_mask.to(Config.DEVICE)
    c_mask = c_mask.to(Config.DEVICE)

    all_predictions = []
    with torch.no_grad():
        for model in models:
            output = model(x, r_mask, c_mask)
            all_predictions.append(output)

    avg_predictions = torch.stack(all_predictions).mean(dim=0)
    return avg_predictions.cpu().numpy()[0]
