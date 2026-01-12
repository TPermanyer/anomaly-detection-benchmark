from .base import AnomalyDetector, ensure_2d
from pyod.models.knn import KNN as PyOD_KNN

class KNN(AnomalyDetector):
    def __init__(self, contamination=0.1, n_neighbors=5, method='largest'):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.method = method  # 'largest', 'mean', 'median'
        self.model = None

    def fit(self, X, y=None):
        X = ensure_2d(X)
        self.model = PyOD_KNN(
            contamination=self.contamination,
            n_neighbors=self.n_neighbors,
            method=self.method
        )
        self.model.fit(X)
        return self

    def score(self, X):
        X = ensure_2d(X)
        return self.model.decision_function(X)
