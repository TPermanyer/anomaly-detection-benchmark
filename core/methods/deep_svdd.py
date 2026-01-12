from .base import AnomalyDetector, ensure_2d
import numpy as np

class DeepSVDD(AnomalyDetector):
    def __init__(self, contamination=0.1, epochs=10, batch_size=32, hidden_neurons=[64, 32]):
        self.contamination = contamination
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_neurons = hidden_neurons
        self.model = None

    def fit(self, X, y=None):
        X = ensure_2d(X)
        from pyod.models.deep_svdd import DeepSVDD as PyOD_DeepSVDD
        n_features = X.shape[1]
        self.model = PyOD_DeepSVDD(
            n_features=n_features,
            contamination=self.contamination,
            epochs=self.epochs,
            batch_size=self.batch_size,
            hidden_neurons=self.hidden_neurons,
            verbose=0
        )
        self.model.fit(X)
        return self

    def score(self, X):
        X = ensure_2d(X)
        return self.model.decision_function(X)
