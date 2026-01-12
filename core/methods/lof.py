from __future__ import annotations
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from .base import AnomalyDetector, ensure_2d


class LOF(AnomalyDetector):
	def __init__(self, n_neighbors=20, contamination="auto"):
		# novelty=True allows scoring on any X via score_samples
		self.model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)


	def fit(self, X, y=None):
		X = ensure_2d(np.asarray(X))
		self.model.fit(X)
		return self


	def score(self, X):
		X = ensure_2d(np.asarray(X))
		# LOF: higher score_samples => more normal → invert
		return -self.model.score_samples(X)