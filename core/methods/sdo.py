from __future__ import annotations
from pyod.models.base import BaseDetector
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

class SDO(BaseDetector):
    """
    SDO (Sparse Data Observers)
    
    An unsupervised anomaly detection method that models the "normal" data mass 
    using a set of representative "observers" (centroids). Anomalies are detected 
    based on their distance to the nearest observer. Points that are far from all 
    observers are considered anomalies.
    
    This implementation uses K-Means centroids as the observers.
    """
    def __init__(self, n_observers=20, contamination=0.1):
        super().__init__(contamination=contamination)
        self.n_observers = n_observers
        self.observers_ = None

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods."""
        # Use MiniBatchKMeans for efficiency (linear complexity O(n))
        self.kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_observers, 
            random_state=42, 
            n_init="auto",
            batch_size=256
        )
        print(f"[SDO] Fitting with {self.n_observers} observers on {X.shape[0]} samples...")
        self.kmeans_.fit(X)
        print(f"[SDO] KMeans converged. Inertia: {self.kmeans_.inertia_:.2f}")
        self.observers_ = self.kmeans_.cluster_centers_
        
        # Decision scores: Distance to nearest observer
        # transform() returns distance to all clusters. We want min.
        # However, transform() is squared Euclidean distance usually? 
        # No, sklearn transform returns Euclidean distance.
        dist = self.kmeans_.transform(X)
        min_dist = dist.min(axis=1)
        
        self.decision_scores_ = min_dist
        self._process_decision_scores()
        print(f"[SDO] Fit complete. Max score: {self.decision_scores_.max():.4f}")
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector."""
        dist = self.kmeans_.transform(X)
        return dist.min(axis=1)

    def score(self, X, y=None):
        return self.decision_function(X)
