from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.decomposition import TruncatedSVD
from .base import AnomalyDetector

_NUM_KINDS = set("biufc")

class SparseAutoPreprocessor(AnomalyDetector):
    """Generic sparse pipeline (no leakage in CV):
    DataFrame -> (numeric: impute/clip/log1p) + (categorical: OHE with cardinality limit) -> hstack(sparse)
               -> TruncatedSVD -> optional Quantile/Power -> RobustScaler -> base detector.
    """
    def __init__(self,
                 base_factory,
                 max_ohe_card: int = 50,
                 use_quantile: str | None = None,   # "normal" | "uniform" | None (applied after SVD)
                 use_power: bool = False,           # Yeo-Johnson (exclusive with quantile)
                 svd_components: int = 100,
                 clip_q: tuple[float, float] | None = (0.001, 0.999),
                 robust_scale: bool = True):
        self.base_factory = base_factory
        self.max_ohe_card = max_ohe_card
        self.use_quantile = use_quantile
        self.use_power = use_power
        self.svd_components = svd_components
        self.clip_q = clip_q
        self.robust_scale = robust_scale

        # fitted artifacts
        self.num_cols: list[str] | None = None
        self.cat_cols: list[str] | None = None
        self.imp_num = None
        self.lo_ = None
        self.hi_ = None
        self.log_mask_ = None
        self.ohe = None
        self.svd = None
        self.qt = None
        self.pt = None
        self.scaler = None
        self.model = None

    # ---------- helpers ----------
    def _split_cols(self, df: pd.DataFrame):
        num, cat = [], []
        for c in df.columns:
            (num if df[c].dtype.kind in _NUM_KINDS else cat).append(c)
        return num, cat

    def _fit_num(self, Xn: np.ndarray):
        if Xn.size == 0:
            return Xn
        self.imp_num = SimpleImputer(strategy="median")
        Xn = self.imp_num.fit_transform(Xn)
        if self.clip_q is not None:
            lo, hi = np.nanquantile(Xn, self.clip_q, axis=0)
            self.lo_, self.hi_ = lo, hi
            Xn = np.clip(Xn, lo, hi)
        mins = Xn.min(axis=0)
        skew = pd.DataFrame(Xn).skew(axis=0).to_numpy()
        self.log_mask_ = (mins >= 0) & (np.abs(skew) > 1.0)
        if self.log_mask_.any():
            Xn[:, self.log_mask_] = np.log1p(Xn[:, self.log_mask_])
        return Xn

    def _transform_num(self, Xn: np.ndarray):
        if Xn.size == 0:
            return Xn
        Xn = self.imp_num.transform(Xn)
        if self.lo_ is not None:
            Xn = np.clip(Xn, self.lo_, self.hi_)
        if self.log_mask_ is not None and self.log_mask_.any():
            Xn[:, self.log_mask_] = np.log1p(np.maximum(Xn[:, self.log_mask_], 0))
        return Xn

    def _fit_cat(self, Xc: pd.DataFrame):
        if Xc.shape[1] == 0:
            return sparse.csr_matrix((Xc.shape[0], 0))
        low, high = [], []
        for c in Xc.columns:
            (low if Xc[c].nunique(dropna=True) <= self.max_ohe_card else high).append(c)
        # One-Hot on low-card
        try:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
        Xo = self.ohe.fit_transform(Xc[low]) if low else sparse.csr_matrix((Xc.shape[0], 0))
        # NOTE: high-card ignored by default; add hashing here if desired.
        return Xo

    def _transform_cat(self, Xc: pd.DataFrame):
        if self.ohe is None or Xc.shape[1] == 0:
            return sparse.csr_matrix((len(Xc), 0))
        # ensure common columns exist
        common = [c for c in Xc.columns if c in getattr(self.ohe, 'feature_names_in_', [])]
        Xc = Xc[common] if common else Xc.iloc[:, :0]
        return self.ohe.transform(Xc) if common else sparse.csr_matrix((len(Xc), 0))

    def _fit_reduce_scale(self, Xs: sparse.csr_matrix):
        n_comp = min(self.svd_components, max(1, min(Xs.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        Z = self.svd.fit_transform(Xs)
        if self.use_quantile is not None:
            self.qt = QuantileTransformer(output_distribution=self.use_quantile, subsample=2000, random_state=42)
            Z = self.qt.fit_transform(Z)
        elif self.use_power:
            self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
            Z = self.pt.fit_transform(Z)
        if self.robust_scale:
            self.scaler = RobustScaler()
            Z = self.scaler.fit_transform(Z)
        return Z

    def _transform_reduce_scale(self, Xs: sparse.csr_matrix):
        Z = self.svd.transform(Xs)
        if self.qt is not None:
            Z = self.qt.transform(Z)
        elif self.pt is not None:
            Z = self.pt.transform(Z)
        if self.scaler is not None:
            Z = self.scaler.transform(Z)
        return Z

    # ---------- API ----------
    def fit(self, X_df: pd.DataFrame, y=None):
        assert isinstance(X_df, pd.DataFrame), "SparseAutoPreprocessor expects a pandas DataFrame"
        self.num_cols, self.cat_cols = self._split_cols(X_df)
        Xn = self._fit_num(X_df[self.num_cols].to_numpy()) if self.num_cols else np.zeros((len(X_df), 0))
        Xc = self._fit_cat(X_df[self.cat_cols]) if self.cat_cols else sparse.csr_matrix((len(X_df), 0))
        Xs = sparse.hstack([sparse.csr_matrix(Xn), Xc], format="csr")
        Z = self._fit_reduce_scale(Xs)

        self.model = self.base_factory()
        try:
            self.model.fit(Z, y)
        except TypeError:
            self.model.fit(Z)
        return self

    def score(self, X_df: pd.DataFrame):
        Xn = self._transform_num(X_df[self.num_cols].to_numpy()) if self.num_cols else np.zeros((len(X_df), 0))
        Xc = self._transform_cat(X_df[self.cat_cols]) if self.cat_cols else sparse.csr_matrix((len(X_df), 0))
        Xs = sparse.hstack([sparse.csr_matrix(Xn), Xc], format="csr")
        Z = self._transform_reduce_scale(Xs)
        return self.model.score(Z)