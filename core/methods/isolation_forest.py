import numpy as np
from sklearn.ensemble import IsolationForest
from .base import AnomalyDetector, ensure_2d

class IForest(AnomalyDetector):
    def __init__(self, n_estimators=200, max_samples="auto", contamination="auto", random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, X, y=None):
        X = ensure_2d(np.asarray(X))
        self.model.fit(X)
        return self

    def score(self, X):
        X = ensure_2d(np.asarray(X))
        # sklearn: higher score_samples = more normal → invert so higher = more anomalous
        return -self.model.score_samples(X)