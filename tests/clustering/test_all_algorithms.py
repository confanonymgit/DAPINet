from sklearn.datasets import make_blobs

from dapinet.clustering.experiment import ALGO_ORDER, run_experiment


def test_all_algorithms_execution():
    """
    Verify that every algorithm in ALGO_ORDER can be executed via run_experiment
    without crashing and returns a valid ARI score.
    """
    # Generate a simple synthetic dataset
    X, y = make_blobs(n_samples=50, centers=3, n_features=2, random_state=42)

    # Run experiment for all algorithms
    # We use n_trials_per_algo=1 and a small timeout to keep it fast
    results = run_experiment(
        X=X,
        y=y,
        n_trials_per_algo=1,
        seed=42,
        store_predictions=False,
        algorithms=ALGO_ORDER,
        optuna_n_jobs=1,
    )

    # Check results for each algorithm
    for algo_name in ALGO_ORDER:
        assert algo_name in results, f"Algorithm {algo_name} missing from results"
        res = results[algo_name]

        # Check basic properties
        assert res.name == algo_name
        assert isinstance(res.best_ari, float)
        assert -1.0 <= res.best_ari <= 1.0, f"Invalid ARI {res.best_ari} for {algo_name}"
        assert isinstance(res.best_params, dict)

        # n_trials should be at least 1
        assert res.n_trials >= 1
