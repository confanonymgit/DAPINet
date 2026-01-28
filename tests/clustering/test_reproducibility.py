import numpy as np
from sklearn.datasets import make_blobs

from dapinet.clustering.experiment import ALGO_ORDER, run_experiment


def test_clustering_reproducibility():
    # Generate synthetic data
    X, y = make_blobs(n_samples=50, centers=3, n_features=2, random_state=42)

    # Define algorithms to test: use all available algorithms
    algorithms = ALGO_ORDER

    # Run experiment 1
    results1 = run_experiment(
        X=X,
        y=y,
        n_trials_per_algo=10,  # Keep it small for speed, but non-zero
        seed=42,
        store_predictions=True,
        algorithms=algorithms,
    )

    # Run experiment 2
    results2 = run_experiment(
        X=X,
        y=y,
        n_trials_per_algo=10,
        seed=42,
        store_predictions=True,
        algorithms=algorithms,
    )

    # Compare results
    for algo in algorithms:
        if algo not in results1 or algo not in results2:
            assert algo in results1, f"Algorithm {algo} missing from results1"
            assert algo in results2, f"Algorithm {algo} missing from results2"

        res1 = results1[algo]
        res2 = results2[algo]

        assert res1.best_ari == res2.best_ari, (
            f"ARI mismatch for {algo}: {res1.best_ari} != {res2.best_ari}"
        )
        assert res1.best_params == res2.best_params, (
            f"Params mismatch for {algo}: {res1.best_params} != {res2.best_params}"
        )

        if res1.prediction is not None and res2.prediction is not None:
            np.testing.assert_array_equal(
                res1.prediction, res2.prediction, err_msg=f"Prediction mismatch for {algo}"
            )
