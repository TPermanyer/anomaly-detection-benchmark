
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from core.metrics.ranking import average_precision

def _recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int):
    if y_true is None or k <= 0:
        return float("nan"), 0
    y_arr = np.asarray(y_true)
    if y_arr.ndim != 1:
        y_arr = y_arr.ravel()
    positives = int(np.sum(y_arr))
    if positives == 0:
        return float("nan"), 0
    idx = np.argsort(scores)[::-1][:min(k, len(scores))]
    found = int(np.sum(y_arr[idx]))
    return (found / positives), found

def _orient_scores(y_true: np.ndarray | None, scores: np.ndarray):
    if y_true is None:
        return scores
    try:
        ap_pos = average_precision(y_true, scores)
        ap_neg = average_precision(y_true, -scores)
        return (-scores) if (np.nan_to_num(ap_neg) > np.nan_to_num(ap_pos)) else scores
    except Exception:
        return scores

def _confusion_matrix_at_k(y_true: np.ndarray, scores: np.ndarray, k: int):
    """Returns TP, FP, TN, FN when taking top-k as predictions"""
    if y_true is None or k <= 0:
        return None
    y_arr = np.asarray(y_true).ravel()
    n = len(y_arr)
    idx_top_k = np.argsort(scores)[::-1][:min(k, n)]
    y_pred = np.zeros(n, dtype=int)
    y_pred[idx_top_k] = 1
    
    tp = int(np.sum((y_pred == 1) & (y_arr == 1)))
    fp = int(np.sum((y_pred == 1) & (y_arr == 0)))
    tn = int(np.sum((y_pred == 0) & (y_arr == 0)))
    fn = int(np.sum((y_pred == 0) & (y_arr == 1)))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

def _recall_only(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Utility that returns only the recall value from _recall_at_k."""
    rec, _ = _recall_at_k(y_true, scores, k)
    return rec if not np.isnan(rec) else float("nan")

def compute_all_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Derive ranking and threshold-aware metrics for labeled anomaly detection."""
    metrics = {}
    metrics["AUC_PR"] = average_precision(y_true, scores)
    try:
        metrics["AUC_ROC"] = roc_auc_score(y_true, scores)
    except ValueError:
        metrics["AUC_ROC"] = float("nan")

    percentiles = np.linspace(50, 99, 30)
    best = {"f1": -1.0, "threshold": None, "precision": float("nan"), "recall": float("nan")}
    for p in percentiles:
        thr = np.percentile(scores, p)
        y_pred = (scores >= thr).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        f1 = f1_score(y_true, y_pred)
        if f1 > best["f1"]:
            best.update(
                {
                    "f1": f1,
                    "threshold": thr,
                    "precision": precision_score(y_true, y_pred),
                    "recall": recall_score(y_true, y_pred),
                    "y_pred": y_pred,
                }
            )

    metrics["F1_opt"] = best["f1"] if best["f1"] >= 0 else float("nan")
    metrics["Precision_opt"] = best["precision"]
    metrics["Recall_opt"] = best["recall"]
    metrics["BestThreshold"] = best["threshold"]

    if "y_pred" in best:
        metrics["MCC"] = matthews_corrcoef(y_true, best["y_pred"])
        metrics["BalancedAcc"] = balanced_accuracy_score(y_true, best["y_pred"])
    else:
        metrics["MCC"] = float("nan")
        metrics["BalancedAcc"] = float("nan")

    for pct in (1, 5, 10):
        k = max(1, int(len(scores) * pct / 100))
        rec = _recall_only(y_true, scores, k)
        metrics[f"Recall@{pct}%"] = rec * 100 if not np.isnan(rec) else float("nan")

    return metrics
