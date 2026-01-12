from __future__ import annotations
import numpy as np
from sklearn.svm import OneClassSVM
from .base import AnomalyDetector, ensure_2d


class OCSVM(AnomalyDetector):
	def __init__(self, kernel="rbf", nu=0.5, gamma="scale"):
		self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)


	def fit(self, X, y=None):
		X = ensure_2d(np.asarray(X))
		self.model.fit(X)
		return self


	def score(self, X):
		X = ensure_2d(np.asarray(X))
		# decision_function: positive inliers, negative outliers → invert
		return -self.model.decision_function(X)