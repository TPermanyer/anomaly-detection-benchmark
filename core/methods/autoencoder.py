from .base import AnomalyDetector, ensure_2d
import numpy as np

class AutoEncoder(AnomalyDetector):
    def __init__(self, contamination=0.1, epochs=50, batch_size=32, hidden_neurons=None):
        self.contamination = contamination
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_neurons = hidden_neurons
        self.model = None

    def fit(self, X, y=None):
        X = ensure_2d(X)
        from pyod.models.auto_encoder import AutoEncoder as PyOD_AE
        n_features = X.shape[1]
        # Default architecture: compress to half, then quarter
        if self.hidden_neurons is None:
            self.hidden_neurons = [max(n_features // 2, 4), max(n_features // 4, 2), max(n_features // 4, 2), max(n_features // 2, 4)]
        self.model = PyOD_AE(
            contamination=self.contamination,
            epoch_num=self.epochs,
            batch_size=self.batch_size,
            hidden_neuron_list=self.hidden_neurons,
            verbose=0
        )
        self.model.fit(X)
        return self

    def score(self, X):
        X = ensure_2d(X)
        return self.model.decision_function(X)
