from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .base import AnomalyDetector, ensure_2d

class PCAReconstruction(AnomalyDetector):
    """PCA reconstruction-error detector with robust guards.

    - Accepts n_components as int, float in (0,1], or None.
    - Clamps integer n_components to a valid range based on train data.
    - Falls back to an orthogonal projection if inverse_transform fails.
    """

    def __init__(self, n_components=None, scale=True, random_state=42, svd_solver="auto"):
        self.n_components = n_components
        self.scale = scale
        self.random_state = random_state
        self.svd_solver = svd_solver
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self._resolved_components = None

    def _resolve_n_components(self, X):
        m = min(X.shape[0], X.shape[1])
        nc = self.n_components
        # If provided as string (from UI), try to coerce
        if isinstance(nc, str):
            s = nc.strip()
            if s.lower() == "none":
                nc = None
            else:
                try:
                    nc = int(s) if s.isdigit() else float(s)
                except Exception:
                    nc = None
        if isinstance(nc, int):
            nc = max(1, min(nc, m))
        elif isinstance(nc, float):
            # sklearn interprets (0,1] as variance ratio, else invalid
            if not (0 < nc <= 1.0):
                nc = None
        elif nc is not None:
            nc = None
        return nc

    def fit(self, X, y=None):
        X = ensure_2d(np.asarray(X))
        if self.scale:
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X)
        else:
            Xs = X
        self._resolved_components = self._resolve_n_components(Xs)
        self.pca = PCA(n_components=self._resolved_components, random_state=self.random_state, svd_solver=self.svd_solver)
        self.pca.fit(Xs)
        return self

    def score(self, X):
        if self.pca is None:
            raise RuntimeError("PCAReconstruction not fitted.")
        X = ensure_2d(np.asarray(X))
        Xs = self.scaler.transform(X) if self.scale and self.scaler is not None else X
        try:
            Xr = self.pca.inverse_transform(self.pca.transform(Xs))
        except Exception:
            # Fallback: orthogonal projection using components_
            comps = getattr(self.pca, "components_", None)
            if comps is None:
                raise
            Xr = (Xs @ comps.T) @ comps  # project onto PCA subspace
        err = ((Xs - Xr) ** 2).sum(axis=1)
        return err