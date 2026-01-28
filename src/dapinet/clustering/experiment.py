import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import adjusted_rand_score

from dapinet.utils import apply_standard_scaling

from .algorithms import ClusteringAlgorithms
from .io import load_json, pair_npz_datasets, save_json
from .registry import REGISTRY

logger = logging.getLogger(__name__)

ALGO_ORDER = list(REGISTRY.keys())


@dataclass
class AlgoResult:
    name: str
    best_ari: float
    best_params: dict[str, Any]
    n_trials: int
    prediction: np.ndarray | None
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "best_ari": float(self.best_ari),
            "best_params": dict(self.best_params),
            "n_trials": int(self.n_trials),
            "duration": float(self.duration),
        }


def _fit_predict_with_runner(name: str, cfg: dict[str, Any], X: np.ndarray) -> np.ndarray:
    # do not mutate caller’s cfg
    setup = {name: dict(cfg)}
    runner = ClusteringAlgorithms(setup)
    results = runner.cluster(X)
    return np.asarray(results[name]["prediction"])


def _adjusted_rand_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return adjusted_rand_score(y_true, y_pred)


def _build_objective(
    name: str,
    base_cfg: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    search_fn: Any,
):
    def objective(trial: optuna.Trial) -> float:
        sampled = search_fn(trial) if search_fn else {}
        merged = {**base_cfg, **sampled}
        labels_pred = _fit_predict_with_runner(name, merged, X)
        ari = _adjusted_rand_score(y, labels_pred)
        trial.set_user_attr("ari", float(ari))
        return ari

    return objective


def _stop_at_perfect_score(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
    if trial.value is not None and trial.value == 1.0:
        study.stop()


def tune_algorithm(
    name: str,
    base_cfg: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    seed: int = 42,
    optuna_n_jobs: int = 1,
    timeout: int | None = None,
) -> AlgoResult:
    start_time = time.perf_counter()

    spec = REGISTRY.get(name)
    if not spec:
        raise ValueError(f"Unknown algorithm: {name}")

    search_fn = spec.search_space

    # If no search space is defined, run a single deterministic fit
    if search_fn is None:
        labels_pred = _fit_predict_with_runner(name, base_cfg, X)
        ari = adjusted_rand_score(y, labels_pred)
        duration = time.perf_counter() - start_time
        return AlgoResult(
            name=name,
            best_ari=float(ari),
            best_params=dict(base_cfg),
            n_trials=1,
            prediction=np.asarray(labels_pred),
            duration=duration,
        )

    objective = _build_objective(name, base_cfg, X, y, search_fn)
    sampler = optuna.samplers.TPESampler(seed=seed)

    prev_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    show_bar = sys.stdout.isatty()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        callbacks=[_stop_at_perfect_score],
        n_trials=n_trials,
        n_jobs=optuna_n_jobs,
        show_progress_bar=show_bar,
        timeout=timeout,
    )
    optuna.logging.set_verbosity(prev_verbosity)
    best_params = {**base_cfg, **study.best_params}
    best_pred = _fit_predict_with_runner(name, best_params, X)

    duration = time.perf_counter() - start_time

    return AlgoResult(
        name=name,
        best_ari=float(study.best_value),
        best_params=best_params,
        n_trials=n_trials,
        prediction=np.asarray(best_pred) if best_pred is not None else None,
        duration=duration,
    )


def run_experiment(
    X: np.ndarray,
    y: np.ndarray,
    n_trials_per_algo: int = 50,
    seed: int = 42,
    algorithms: list[str] | None = None,
    optuna_n_jobs: int = 1,
    timeout: int | None = None,
) -> dict[str, AlgoResult]:
    rng = np.random.default_rng(seed)
    n_clusters = len(set(y))

    # Derive minimal base configs internally (n_clusters only where required)
    # ORDER MUST MATCH REGISTRY to ensure consistency with ALGO_ORDER
    default_base: dict[str, dict[str, Any]] = {
        "k-means": {"n_clusters": n_clusters, "random_state": seed},
        "k-medians": {"n_clusters": n_clusters, "random_state": seed},
        "spectral_clustering": {
            "n_clusters": n_clusters,
            "eigen_solver": "arpack",
            "affinity": "nearest_neighbors",
            "random_state": seed,
        },
        "ward": {"n_clusters": n_clusters, "linkage": "ward"},
        "agglomerative": {"n_clusters": n_clusters, "metric": "euclidean", "linkage": "single"},
        "dbscan": {},
        "hdbscan": {},
        "optics": {},
        "birch": {"n_clusters": n_clusters},
        "gaussian": {"n_components": n_clusters, "covariance_type": "full", "random_state": seed},
        "mean_shift": {},
        "affinity_propagation": {"max_iter": 2000, "random_state": seed},
    }

    selected = algorithms or list(default_base.keys())
    algo_results: dict[str, AlgoResult] = {}

    for idx, name in enumerate(selected, start=1):
        base_cfg = default_base.get(name, {})
        logger.info(f"[TUNE {idx}/{len(selected)}] {name}")
        try:
            child_seed = int(rng.integers(0, 1_000_000))
            res = tune_algorithm(
                name,
                base_cfg,
                X,
                y,
                n_trials=n_trials_per_algo,
                seed=child_seed,
                optuna_n_jobs=optuna_n_jobs,
                timeout=timeout,
            )
            algo_results[name] = res
            logger.info(f"  Best ARI: {res.best_ari:.4f}")
        except Exception as e:
            logger.warning(f"Skipped {name} due to error: {e}")
            continue

    return algo_results


def process_single_dataset(
    ds_id: str,
    X_raw: np.ndarray,
    y: np.ndarray,
    n_trials_per_algo: int,
    master_seed: int,
    algorithms: list[str] | None,
    optuna_n_jobs: int,
    timeout: int | None,
) -> tuple[str, dict[str, AlgoResult], np.ndarray, np.ndarray]:
    logger.info(f"=== Dataset: {ds_id} ===")

    X_norm = apply_standard_scaling(X_raw)

    exp = run_experiment(
        X=X_norm,
        y=y,
        n_trials_per_algo=n_trials_per_algo,
        seed=master_seed,
        algorithms=algorithms,
        optuna_n_jobs=optuna_n_jobs,
        timeout=timeout,
    )
    return ds_id, exp, X_norm, y


def run_all_datasets_in_npz(
    npz_path: Path,
    algorithms: list[str] | None = None,
    n_trials_per_algo: int = 50,
    master_seed: int = 42,
    optuna_n_jobs: int = 1,
    timeout: int | None = None,
) -> None:
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")
    sidecar = npz_path.with_suffix(".json")
    meta = load_json(sidecar)
    meta = meta if meta is not None else {}

    with np.load(npz_path) as npz:
        pairs = pair_npz_datasets(npz)
        if not pairs:
            raise ValueError("No (X,y) pairs found in NPZ.")

        predictions: defaultdict[str, dict[str, np.ndarray]] = defaultdict(dict)
        training_data: dict[str, np.ndarray] = {}
        results_root = meta.setdefault("results", {})

        for ds_id, xk, yk in pairs:
            X_raw = np.asarray(npz[xk])
            y = np.asarray(npz[yk])

            _, exp, X_norm, _ = process_single_dataset(
                ds_id, X_raw, y, n_trials_per_algo, master_seed, algorithms, optuna_n_jobs, timeout
            )

            # Collect ARI vector for training data
            ari_vector = []
            for algo in ALGO_ORDER:
                if algo in exp:
                    ari_vector.append(exp[algo].best_ari)
                else:
                    ari_vector.append(-1.0)

            training_data[f"{ds_id}_X"] = X_norm
            training_data[f"{ds_id}_y"] = np.array(ari_vector, dtype=np.float64)

            ds_results = results_root.setdefault(ds_id, {})
            for algo_name, algo_res in exp.items():
                ds_results[algo_name] = {
                    "best_ari": float(algo_res.best_ari),
                    "best_params": dict(algo_res.best_params),
                }
                if algo_res.prediction is not None:
                    predictions[ds_id][algo_name] = np.asarray(algo_res.prediction, dtype=np.int8)

    # Save predictions NPZ
    preds_path = npz_path.with_name(f"{npz_path.stem}_algorithm_predictions.npz")
    np.savez_compressed(preds_path, **predictions)
    logger.info(f"Saved predictions to: {preds_path}")

    # Save training NPZ
    if training_data:
        train_path = npz_path.with_name(f"{npz_path.stem}_training.npz")
        np.savez_compressed(train_path, **training_data)
        logger.info(f"Saved training data to: {train_path}")

    # Save updated JSON sidecar
    save_json(sidecar, meta)
    logger.info(f"Updated results in sidecar JSON: {sidecar}")


def main():
    root = Path(__file__).resolve().parents[3]
    npz_path = root / "datasets" / "synthetic" / "CesarComin" / "cfg000" / "CesarComin_cfg000.npz"
    if npz_path.exists():
        run_all_datasets_in_npz(
            npz_path=npz_path,
            n_trials_per_algo=50,
            master_seed=42,
        )
    else:
        print(f"Example file not found: {npz_path}")


if __name__ == "__main__":
    main()
