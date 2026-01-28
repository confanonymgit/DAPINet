import json
from pathlib import Path
from typing import Any

import numpy as np


def pair_npz_datasets(npz: np.lib.npyio.NpzFile) -> list[tuple[str, str, str]]:
    """Return (dataset_id, x_key, y_key) pairs."""
    keys = set(npz.files)
    pairs: list[tuple[str, str, str]] = []
    for k in sorted(keys):
        if k.endswith("_X"):
            ds_id = k[:-2]
            yk = f"{ds_id}_y"
            if yk in keys:
                pairs.append((ds_id, k, yk))
    return pairs


def save_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
