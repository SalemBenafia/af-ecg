"""Metrics: challenge F1 (mean F1 of Normal, AF, Other), AF sensitivity/specificity, bootstrap CIs."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from .data import LABELS


def summarize(y_true, y_pred, p_af=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    t, p = y_true == "A", y_pred == "A"
    tp, fn, fp, tn = (t & p).sum(), (t & ~p).sum(), (~t & p).sum(), (~t & ~p).sum()
    out = {f"F1_{c}": float(v) for c, v in zip(LABELS, f1)}
    out |= {
        "challenge_F1": float(np.mean(f1[:3])),          # official 2017 score: mean of N, A, O
        "accuracy": float((y_true == y_pred).mean()),
        "AF_sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "AF_specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "AF_PPV": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "n": int(len(y_true)),
    }
    if p_af is not None and 0 < t.sum() < len(t):
        out["AF_AUROC"] = float(roc_auc_score(t, p_af))
    return out


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def bootstrap_ci(y_true, y_pred, metric="challenge_F1", groups=None, n_boot=2000, seed=0, alpha=0.05):
    """Resample recordings (or groups, e.g. AFDB records) with replacement."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)
    ids = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ids}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[g] for g in rng.choice(ids, len(ids), replace=True)])
        v = summarize(y_true[idx], y_pred[idx])[metric]
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def coverage_curve(y_true, y_pred, confidence, thresholds):
    """Abstain when the model's top probability < threshold: % kept vs score on the kept ones."""
    y_true, y_pred, confidence = map(np.asarray, (y_true, y_pred, confidence))
    rows = []
    for th in thresholds:
        keep = confidence >= th
        s = summarize(y_true[keep], y_pred[keep]) if keep.sum() > 10 else {}
        rows.append({"threshold": float(th), "kept_pct": float(keep.mean() * 100),
                     "challenge_F1": s.get("challenge_F1", float("nan")),
                     "AF_sensitivity": s.get("AF_sensitivity", float("nan"))})
    return rows
