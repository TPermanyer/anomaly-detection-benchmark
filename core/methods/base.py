from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class AnomalyDetector(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        ...

    @abstractmethod
    def score(self, X) -> np.ndarray:
        """Return anomaly scores where higher = more anomalous."""
        ...


def ensure_2d(X):
    import numpy as np
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X