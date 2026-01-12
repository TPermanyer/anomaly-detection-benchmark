
from .base import AnomalyDetector, ensure_2d
from pyod.models.ecod import ECOD as PyOD_ECOD

class ECOD(AnomalyDetector):
    def __init__(self, contamination=0.1, n_jobs=1):
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X, y=None):
        X = ensure_2d(X)
        self.model = PyOD_ECOD(contamination=self.contamination, n_jobs=self.n_jobs)
        self.model.fit(X)
        return self

    def score(self, X):
        X = ensure_2d(X)
        return self.model.decision_function(X)
