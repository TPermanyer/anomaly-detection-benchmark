import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def average_precision(y_true, scores) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    return float(average_precision_score(y_true, scores))


def precision_recall_points(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    prec, rec, thr = precision_recall_curve(y_true, scores)
    return prec, rec, thr


def precision_at_k(y_true, scores, k: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    k = int(max(1, min(k, len(scores))))
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].sum() / k)