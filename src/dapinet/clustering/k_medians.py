import numpy as np
from scipy.spatial.distance import cdist

from dapinet.clustering.base import ClusteringAlgorithm


class KMedians(ClusteringAlgorithm):
    """K-Medians with multiple inits and L1 distance."""

    def __init__(
        self,
        n_clusters: int,
        n_init: int = 100,
        max_iter: int = 300,
        verbose: bool = False,
        random_state: int = 0,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.max_iter = max_iter
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        """Fit medians to data."""
        np.random.seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        best_median = None
        best_labels = None
        best_inertia = np.inf

        for init_num in range(self.n_init):
            # Randomly initialize medians
            medians = data[rng.choice(data.shape[0], self.n_clusters, replace=False)]
            labels = np.zeros(data.shape[0], dtype=int)

            for iteration in range(self.max_iter):
                ## Assign each point to the nearest median using L1 distance
                distances = cdist(data, medians, metric="cityblock")
                new_labels = np.argmin(distances, axis=1)

                # Update medians
                new_medians = []
                for i in range(self.n_clusters):
                    if np.any(new_labels == i):
                        cluster_data = data[new_labels == i]
                        median = np.median(cluster_data, axis=0)
                    else:
                        # Handle empty cluster: reinitialize to random point
                        median = data[rng.integers(data.shape[0])]
                    new_medians.append(median)
                new_medians = np.array(new_medians)

                if np.all(new_labels == labels):
                    break

                medians = new_medians
                labels = new_labels

            # Calculate total L1-distance (inertia)
            # We can reuse cdist here but we need distance to assigned center
            # Or just sum the min distances
            final_distances = cdist(data, medians, metric="cityblock")
            inertia = np.sum(np.min(final_distances, axis=1))

            if self.verbose:
                print(f"[Init {init_num}] Inertia: {inertia:.4f}")

            if inertia < best_inertia:
                best_median = medians
                best_labels = labels
                best_inertia = inertia

        self.cluster_centers_ = best_median
        self.labels_ = best_labels

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Assign clusters for new data."""
        if self.cluster_centers_ is None:
            raise Exception("Model has not been fitted yet.")
        distances = cdist(data, self.cluster_centers_, metric="cityblock")
        return np.argmin(distances, axis=1)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """Fit and return labels."""
        self.fit(data)
        return self.labels_
