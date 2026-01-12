
import streamlit as st
import pandas as pd
import numpy as np
import scipy.io
import io
import tempfile
import h5py
import os
import sys

# Ensure ROOT is in path for core imports if needed, though this is usually handled by the main script
# We will assume core is importable
from core.datasets.io import parse_dataframe

def downsample_dataframe(df: pd.DataFrame, n_max: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= n_max:
        return df
    return df.sample(n=n_max, random_state=random_state)


def _parse_literal(val):
    if isinstance(val, (int, float, bool)) or val is None:
        return val
    s = str(val).strip()
    if s.lower() == "none":
        return None
    try:
        if any(ch in s.lower() for ch in [".", "e"]):
            return float(s)
        return int(s)
    except Exception:
        return s

def _read_mat_to_df(file_bytes):
    # Try scipy.io.loadmat first
    try:
        mat = scipy.io.loadmat(io.BytesIO(file_bytes))
    except Exception:
        # Fallback for v7.3 mat files (HDF5)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            with h5py.File(tmp_path, 'r') as f:
                mat = {}
                for k, v in f.items():
                    if isinstance(v, h5py.Dataset):
                        val = np.array(v)
                        # HDF5 from MATLAB is often transposed
                        if val.ndim == 2:
                            val = val.T
                        mat[k] = val
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise ValueError("Could not read .mat file (tried scipy.io and h5py)")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Heuristics to find X and y
    X, y = None, None
    keys = [k for k in mat.keys() if not k.startswith('__')]
    
    # Find X
    for k in ['X', 'data', 'features']:
        if k in keys and isinstance(mat[k], np.ndarray):
            X = mat[k]
            break
    
    if X is None:
        # Look for largest 2D array
        candidates = [mat[k] for k in keys if isinstance(mat[k], np.ndarray) and mat[k].ndim == 2]
        if candidates:
            X = max(candidates, key=lambda arr: arr.size)
            
    if X is None:
        raise ValueError("Could not identify data matrix in .mat file.")

    # Find y
    for k in ['y', 'label', 'labels', 'ground_truth']:
        if k in keys and isinstance(mat[k], np.ndarray):
            y = mat[k]
            break
            
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    
    if y is not None:
        y = y.ravel()
        if len(y) == len(df):
            df["label"] = y
            
    return df

@st.cache_data(show_spinner=False)
def load_and_parse_data(file_bytes, file_name, feature_cols, label_col, timestamp_col):
    """Cached data loading and parsing"""
    if file_name.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(file_bytes))
    elif file_name.endswith(".mat"):
        df = _read_mat_to_df(file_bytes)
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))
    
    X, y, meta = parse_dataframe(df, feature_cols=feature_cols,
                                 label_col=(None if label_col == "(none)" else label_col),
                                 timestamp_col=(None if timestamp_col == "(none)" else timestamp_col))
    return df, X, y, meta
