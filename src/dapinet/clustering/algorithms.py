from typing import Any

import numpy as np
from sklearn.neighbors import kneighbors_graph

from .registry import REGISTRY


class ClusteringAlgorithmFactory:
    def create_algorithm(self, algorithm_name: str, config: dict) -> Any:
        if algorithm_name not in REGISTRY:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        spec = REGISTRY[algorithm_name]
        return spec.cls(**config)


class ClusteringAlgorithms:
    def __init__(self, base_config: dict[str, dict]):
        self.algorithm_factory = ClusteringAlgorithmFactory()
        self.base_config = base_config

    def cluster(self, data: np.ndarray) -> dict[str, dict]:
        results = {}
        for name, cfg in self.base_config.items():
            # Create a copy of config to avoid side effects
            config = cfg.copy()
            algorithm = self.creator(data, name, config)
            prediction = self.predictor(algorithm, data)
            results[name] = {"config": cfg, "prediction": np.asarray(prediction)}
        return results

    def creator(self, data: np.ndarray, algorithm_name: str, config: dict) -> Any:
        # Handle connectivity graph for spectral/agglomerative if needed
        if "n_neighbors" in config and algorithm_name in ["ward", "agglomerative"]:
            config["connectivity"] = self.get_connectivity_graph(data, config["n_neighbors"])
            del config["n_neighbors"]

        return self.algorithm_factory.create_algorithm(algorithm_name, config)

    def predictor(self, algorithm, data):
        if hasattr(algorithm, "fit_predict"):
            return algorithm.fit_predict(data)
        else:
            algorithm.fit(data)
            if hasattr(algorithm, "labels_"):
                return algorithm.labels_
            return algorithm.predict(data)

    @staticmethod
    def get_connectivity_graph(data, n_neighbors):
        if n_neighbors is None:
            return None
        else:
            connectivity = kneighbors_graph(data, n_neighbors=n_neighbors, include_self=False)
            return 0.5 * (connectivity + connectivity.T)
