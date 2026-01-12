# core/methods/ensemble.py
import numpy as np
from .base import AnomalyDetector, ensure_2d

def _percentile(x):
    # ranks in [0,1]
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x)+1, dtype=float)
    return ranks / len(x)

class MeanPercentileEnsemble(AnomalyDetector):
    def __init__(self, factories):
        self.factories = factories
        self.models = []
        self._flip = None  # NUEVO: orientación por modelo (-1 si hay que invertir)

    def fit(self, X, y=None):
        X = ensure_2d(np.asarray(X))
        self.models = [f() for f in self.factories]
        for m in self.models:
            m.fit(X)
        # NUEVO: determinar orientación si hay etiquetas
        self._flip = [1] * len(self.models)
        if y is not None:
            yb = np.asarray(y).ravel().astype(float)
            if np.nanstd(yb) > 0:
                for j, m in enumerate(self.models):
                    s = np.asarray(m.score(X))
                    if np.nanstd(s) == 0:
                        self._flip[j] = 1
                        continue
                    # correlación simple para decidir orientación: si es negativa, invertir
                    try:
                        corr = np.corrcoef(np.nan_to_num(s), np.nan_to_num(yb))[0, 1]
                    except Exception:
                        corr = 0.0
                    self._flip[j] = -1 if (corr < 0) else 1
        return self

    def score(self, X):
        X = ensure_2d(np.asarray(X))
        # aplicar orientación y combinar por percentiles
        S_list = []
        for j, m in enumerate(self.models):
            s = np.asarray(m.score(X))
            flip = 1 if (self._flip is None or j >= len(self._flip)) else self._flip[j]
            s = flip * s  # mayor => más anómalo
            S_list.append(s)
        S = np.column_stack(S_list)
        P = np.column_stack([_percentile(S[:, j]) for j in range(S.shape[1])])
        return np.nanmean(P, axis=1)
