from __future__ import annotations
import numpy as np
from typing import List, Tuple, Callable
from sklearn.model_selection import KFold, StratifiedKFold
from ..metrics.ranking import average_precision
import warnings

ProgressCB = Callable[[int], None]


def _slice_rows(X, idx):
    # Works for np.ndarray and pandas.DataFrame/Series
    try:
        import pandas as pd  # lazy import to avoid hard dep here
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.iloc[idx]
    except Exception:
        pass
    return X[idx]


def kfold_indices(n: int, y=None, n_splits: int = 5, random_state: int = 42):
    if y is not None:
        y_arr = np.asarray(y)
        if len(np.unique(y_arr)) == 2 and 0 < y_arr.sum() < len(y_arr):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return list(skf.split(np.arange(n), y_arr))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(np.arange(n)))


def cv_average_precision(model_factory,
                         X,
                         y,
                         n_splits: int = 5,
                         random_state: int = 42,
                         progress_cb: ProgressCB | None = None) -> Tuple[float, List[float]]:
    if y is None:
        return float("nan"), []
    y_arr = np.asarray(y)
    # Select splitter based on data type
    splits = kfold_indices(len(X), y_arr, n_splits=n_splits, random_state=random_state)
    aps: List[float] = []
    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        Xtr = _slice_rows(X, tr_idx)
        Xte = _slice_rows(X, te_idx)
        yte = y_arr[te_idx]
        try:
            model = model_factory()
            # supervised: fit(Xtr, y_train); unsupervised: fit(Xtr)
            try:
                model.fit(Xtr, y_arr[tr_idx])
            except TypeError:
                model.fit(Xtr)
            scores = model.score(Xte)
            ap = average_precision(yte, scores)
        except Exception as e:
            warnings.warn(f"cv fold {fold} failed: {e}")
            ap = np.nan
        aps.append(ap)
        if progress_cb is not None:
            progress_cb(1)
    mean_ap = float(np.nanmean(aps)) if len(aps) else float("nan")
    return mean_ap, aps