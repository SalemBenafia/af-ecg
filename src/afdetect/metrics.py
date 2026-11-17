"""Metrics: challenge F1, per-class scores, bootstrap CIs and abstain coverage."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from .data import LABELS

NAN = float("nan")


def _safe_div(num, den):
    """num / den as a float, NaN when the denominator is 0."""
    return float(num / den) if den else NAN


def summarize(y_true, y_pred, p_af=None):
    """All headline metrics for one set of predictions.

    challenge_F1 = mean F1 of N, A and O (official 2017 score; ~ excluded). AF metrics treat A as positive
    and every other class as negative. AF_AUROC needs p_af (probability of AF) and both outcomes present."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    t, p = y_true == "A", y_pred == "A"
    tp, fn, fp, tn = int((t & p).sum()), int((t & ~p).sum()), int((~t & p).sum()), int((~t & ~p).sum())
    out = {f"F1_{c}": float(v) for c, v in zip(LABELS, f1)}
    out |= {
        "challenge_F1": float(np.mean(f1[:3])),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else NAN,
        "AF_sensitivity": _safe_div(tp, tp + fn),
        "AF_specificity": _safe_div(tn, tn + fp),
        "AF_PPV": _safe_div(tp, tp + fp),
        "AF_AUROC": NAN,
        "n": int(len(y_true)),
    }
    if p_af is not None and 0 < t.sum() < len(t):
        out["AF_AUROC"] = float(roc_auc_score(t, np.asarray(p_af)))
    return out


def confusion(y_true, y_pred):
    """4x4 confusion matrix: rows = true class, columns = predicted class, both in LABELS order."""
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def bootstrap_ci(y_true, y_pred, metric="challenge_F1", groups=None, n_boot=2000, seed=0, p_af=None, alpha=0.05):
    """Percentile bootstrap (1 - alpha) CI of one summarize() metric.

    Resamples recordings with replacement, or whole groups when `groups` is given (e.g. AFDB windows from
    the same person stay together). Resamples where the metric is undefined are skipped."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p_af = None if p_af is None else np.asarray(p_af)
    groups = np.arange(len(y_true)) if groups is None else np.asarray(groups)
    ids = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ids}
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(ids, len(ids), replace=True)])
        v = summarize(y_true[idx], y_pred[idx], None if p_af is None else p_af[idx])[metric]
        if np.isfinite(v):
            values.append(v)
    if not values:
        return NAN, NAN
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def coverage_curve(y_true, y_pred, confidence, thresholds, min_kept=11):
    """Abstain when confidence < threshold: for each threshold, how many records are kept and the scores on them.

    Scores are NaN when fewer than `min_kept` records are kept."""
    y_true, y_pred, confidence = map(np.asarray, (y_true, y_pred, confidence))
    rows = []
    for th in thresholds:
        keep = confidence >= th
        s = summarize(y_true[keep], y_pred[keep]) if keep.sum() >= min_kept else {}
        rows.append({"threshold": float(th), "n_kept": int(keep.sum()),
                     "kept_pct": float(100 * keep.mean()) if len(keep) else NAN,
                     "challenge_F1": s.get("challenge_F1", NAN),
                     "AF_sensitivity": s.get("AF_sensitivity", NAN),
                     "accuracy": s.get("accuracy", NAN)})
    return rows
