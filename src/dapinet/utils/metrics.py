from collections.abc import Callable

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def dunn_score(data: np.ndarray, labels: np.ndarray, metric: str | Callable = "euclidean") -> float:
    """Compute Dunn index for clustering labels."""
    distances = pairwise_distances(data, metric=metric)
    return _calculate_dunn_from_distances(labels, distances)


def normalize_to_smallest_integers(labels):
    """Normalize labels to consecutive integers."""
    unique_labels = np.unique(labels)
    return np.searchsorted(unique_labels, labels)


def _calculate_dunn_from_distances(labels, distances):
    """Compute Dunn index from a distance matrix."""
    labels = normalize_to_smallest_integers(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        raise ValueError("Dunn Index is undefined for fewer than 2 clusters.")

    if distances.shape[0] != len(labels):
        raise ValueError("Labels and distance matrix shape mismatch.")

    # Min inter-cluster distances
    min_intercluster = np.inf
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                dist = distances[i, j]
                if dist < min_intercluster:
                    min_intercluster = dist

    # Max intra-cluster distance (cluster diameters)
    max_diameter = 0.0
    for cluster_id in unique_labels:
        idx = np.where(labels == cluster_id)[0]
        if len(idx) < 2:
            continue
        intra_dists = distances[np.ix_(idx, idx)]
        local_diameter = np.max(intra_dists)
        if local_diameter > max_diameter:
            max_diameter = local_diameter

    if max_diameter == 0:
        raise ValueError("Max diameter is zero. Likely all clusters are singletons.")

    return min_intercluster / max_diameter


# Example usage:
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    score = dunn_score(X, labels)
    print(score)  # Should print a Dunn index value, 9.51
