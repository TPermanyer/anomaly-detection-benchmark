from __future__ import annotations
import numpy as np
from pyod.models.hbos import HBOS as _HBOS
from .base import AnomalyDetector, ensure_2d


class HBOS(AnomalyDetector):
	def __init__(self, n_bins=10, alpha=0.1, tol=0.5):
		self.model = _HBOS(n_bins=n_bins, alpha=alpha, tol=tol)


	def fit(self, X, y=None):
		X = ensure_2d(np.asarray(X))
		self.model.fit(X)
		return self


	def score(self, X):
		X = ensure_2d(np.asarray(X))
		# PyOD: decision_function → higher = more anomalous (already correct)
		return self.model.decision_function(X)