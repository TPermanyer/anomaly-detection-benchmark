import numpy as np
import pandas as pd
from typing import Optional, Tuple

NUMERIC_KINDS = set(list("biufc"))  # bool, int, unsigned, float, complex


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # one-hot encode categorical columns, keep numeric as-is
    non_numeric = [c for c in df.columns if df[c].dtype.kind not in NUMERIC_KINDS]
    if non_numeric:
        df = pd.get_dummies(df, columns=non_numeric, dummy_na=False)
    return df


def parse_dataframe(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
):
    meta = df.copy()

    if label_col is not None and label_col in df.columns:
        y = df[label_col].astype(int).to_numpy()
    else:
        y = None

    X = df[feature_cols].copy()
    X = _ensure_numeric(X)
    X = X.to_numpy()

    if timestamp_col is not None and timestamp_col in df.columns:
        # Keep original ordering but warn if unsorted; MVP keeps it simple
        try:
            pd.to_datetime(df[timestamp_col])
        except Exception:
            pass  # Streamlit page shows errors if needed in future

    return X, y, meta