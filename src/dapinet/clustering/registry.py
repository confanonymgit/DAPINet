from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture

from .k_medians import KMedians
from .search_spaces import SEARCH_SPACES


@dataclass
class AlgorithmSpec:
    cls: type[Any]
    search_space: Callable[[optuna.Trial], dict[str, Any]] | None


REGISTRY: dict[str, AlgorithmSpec] = {
    "k-means": AlgorithmSpec(KMeans, SEARCH_SPACES["k-means"]),
    "k-medians": AlgorithmSpec(KMedians, SEARCH_SPACES["k-medians"]),
    "spectral_clustering": AlgorithmSpec(SpectralClustering, SEARCH_SPACES["spectral_clustering"]),
    "ward": AlgorithmSpec(AgglomerativeClustering, SEARCH_SPACES["ward"]),
    "agglomerative": AlgorithmSpec(AgglomerativeClustering, SEARCH_SPACES["agglomerative"]),
    "dbscan": AlgorithmSpec(DBSCAN, SEARCH_SPACES["dbscan"]),
    "hdbscan": AlgorithmSpec(HDBSCAN, SEARCH_SPACES["hdbscan"]),
    "optics": AlgorithmSpec(OPTICS, SEARCH_SPACES["optics"]),
    "birch": AlgorithmSpec(Birch, SEARCH_SPACES["birch"]),
    "gaussian": AlgorithmSpec(GaussianMixture, SEARCH_SPACES["gaussian"]),
    "mean_shift": AlgorithmSpec(MeanShift, SEARCH_SPACES["mean_shift"]),
    "affinity_propagation": AlgorithmSpec(
        AffinityPropagation, SEARCH_SPACES["affinity_propagation"]
    ),
}
