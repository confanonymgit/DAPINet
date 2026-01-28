from abc import ABC, abstractmethod

import numpy as np


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        pass
