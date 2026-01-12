# core/methods/preproc_wrapper.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer
from .base import AnomalyDetector, ensure_2d

__all__ = ["PreprocWrapper", "AutoPreprocessor"]  # <- export explícito


class PreprocWrapper(AnomalyDetector):
    """Versión simple: Impute + RobustScaler (opcional), sin transformaciones avanzadas."""
    def __init__(self, base_factory, impute: bool = True, scale: bool = True):
        self.base_factory = base_factory
        self.impute = impute
        self.scale = scale
        self.imputer = None
        self.scaler = None
        self.model: AnomalyDetector | None = None

    def _transform(self, X):
        X = ensure_2d(np.asarray(X))
        if self.imputer is not None:
            X = self.imputer.transform(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def fit(self, X, y=None):
        X = ensure_2d(np.asarray(X))
        if self.impute:
            self.imputer = SimpleImputer(strategy="median")
            X = self.imputer.fit_transform(X)
        if self.scale:
            self.scaler = RobustScaler()
            X = self.scaler.fit_transform(X)
        self.model = self.base_factory()
        self.model.fit(X)
        return self

    def score(self, X):
        if self.model is None:
            raise RuntimeError("PreprocWrapper not fitted")
        X = self._transform(X)
        return self.model.score(X)


class AutoPreprocessor(AnomalyDetector):
    """
    Preprocesado genérico (sin fuga en CV):
      Impute → Clip por cuantiles → (auto log1p) → (Quantile o Power) → RobustScaler
    """
    def __init__(self,
                 base_factory,
                 impute: bool = True,
                 scale: bool = True,
                 clip_quantiles: tuple[float, float] | None = (0.001, 0.999),
                 auto_log1p: bool = True,
                 quantile: str | None = None,   # "normal" | "uniform" | None
                 power: bool = False):          # Yeo-Johnson si True (excluyente con quantile)
        self.base_factory = base_factory
        self.impute = impute
        self.scale = scale
        self.clip_quantiles = clip_quantiles
        self.auto_log1p = auto_log1p
        self.quantile = quantile
        self.power = power

        # fitted artifacts
        self.imputer = None
        self.lo_ = None
        self.hi_ = None
        self.log_mask_ = None
        self.qt = None
        self.pt = None
        self.scaler = None

        self.model: AnomalyDetector | None = None

    # ---------- internal transforms ----------
    def _fit_transforms(self, X: np.ndarray):
        X = ensure_2d(np.asarray(X))

        # 1) Impute
        if self.impute:
            self.imputer = SimpleImputer(strategy="median")
            X = self.imputer.fit_transform(X)

        # 2) Clip (guardar cuantiles para usar en transform)
        if self.clip_quantiles is not None:
            lo_q, hi_q = self.clip_quantiles
            self.lo_ = np.nanquantile(X, lo_q, axis=0)
            self.hi_ = np.nanquantile(X, hi_q, axis=0)
            X = np.clip(X, self.lo_, self.hi_)

        # 3) Auto log1p en columnas positivas y muy sesgadas
        if self.auto_log1p:
            mins = X.min(axis=0)
            skew = pd.DataFrame(X).skew(axis=0).to_numpy()
            self.log_mask_ = (mins >= 0) & (np.abs(skew) > 1.0)
            if self.log_mask_.any():
                X[:, self.log_mask_] = np.log1p(X[:, self.log_mask_])
        else:
            self.log_mask_ = np.zeros(X.shape[1], dtype=bool)

        # 4) Quantile o Power (mutuamente excluyentes)
        if self.quantile is not None:
            self.qt = QuantileTransformer(output_distribution=self.quantile, subsample=2000, random_state=42)
            X = self.qt.fit_transform(X)
        elif self.power:
            self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
            X = self.pt.fit_transform(X)

        # 5) Escalado robusto
        if self.scale:
            self.scaler = RobustScaler()
            X = self.scaler.fit_transform(X)

        return X

    def _transform(self, X: np.ndarray):
        X = ensure_2d(np.asarray(X))
        if self.imputer is not None:
            X = self.imputer.transform(X)
        if self.lo_ is not None and self.hi_ is not None:
            X = np.clip(X, self.lo_, self.hi_)
        if self.log_mask_ is not None and self.log_mask_.any():
            # asegurar no-negatividad antes de log1p en inferencia
            X[:, self.log_mask_] = np.log1p(np.maximum(X[:, self.log_mask_], 0))
        if self.qt is not None:
            X = self.qt.transform(X)
        elif self.pt is not None:
            X = self.pt.transform(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    # ---------- model API ----------
    def fit(self, X, y=None):
        Xp = self._fit_transforms(X)
        self.model = self.base_factory()
        # Supervised models need y; unsupervised will raise TypeError if we pass it
        try:
            self.model.fit(Xp, y)
        except TypeError:
            self.model.fit(Xp)
        return self

    def score(self, X):
        if self.model is None:
            raise RuntimeError("AutoPreprocessor not fitted")
        Xp = self._transform(X)
        return self.model.score(Xp)
