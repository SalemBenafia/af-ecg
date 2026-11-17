"""Models: rr_rule baseline, logreg_rr baseline and the main gbm.

All share fit(X, y), predict_proba(X) and classes_; make_model returns (model, feature columns)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import LABELS
from .features import FEATURES, RR_FEATURES

MODEL_NAMES = ("rr_rule", "logreg_rr", "gbm")


class RRRule:
    """Baseline 1: "AF if normalised RMSSD (beat-to-beat variability) > threshold, else Normal".

    fit() tries the TRAIN quantiles 0.50, 0.51, ..., 0.99 of rmssd_norm and keeps the threshold with the
    best AF F1. A missing rmssd_norm counts as 0 (Normal)."""

    feature = "rmssd_norm"
    classes_ = np.array(["N", "A"])

    def _values(self, X):
        return X[self.feature].fillna(0).to_numpy(dtype=float)

    def fit(self, X, y):
        """Choose the threshold that maximises AF F1 on the training data."""
        x, is_af = self._values(X), np.asarray(y) == "A"
        best_f1, best_th = -1.0, float(np.median(x))
        for th in np.quantile(x, np.round(np.arange(50, 100) / 100, 2)):
            pred = x > th
            tp, fp, fn = (pred & is_af).sum(), (pred & ~is_af).sum(), (~pred & is_af).sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
            if f1 > best_f1:
                best_f1, best_th = float(f1), float(th)
        self.threshold_, self.train_af_f1_ = best_th, best_f1
        return self

    def predict_proba(self, X):
        """Two columns, Normal and AF: 1 for the predicted class, 0 for the other."""
        af = (self._values(X) > self.threshold_).astype(float)
        return np.column_stack([1 - af, af])


def make_model(name, seed=0):
    """Return (unfitted model, feature columns) for "rr_rule", "logreg_rr" or "gbm"."""
    if name == "rr_rule":
        return RRRule(), [RRRule.feature]
    if name == "logreg_rr":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=3000, class_weight="balanced"))
        return model, list(RR_FEATURES)
    if name == "gbm":
        model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, class_weight="balanced",
                                               early_stopping=True, validation_fraction=0.15,
                                               random_state=seed)
        return model, list(FEATURES)
    raise ValueError(f"Unknown model {name!r}: expected one of {MODEL_NAMES}")


def predict(model, cols, X):
    """Predicted labels and an n x 4 probability matrix in LABELS order (0 for classes the model lacks)."""
    proba = model.predict_proba(X[cols])
    classes = list(model.classes_)
    full = np.zeros((len(X), len(LABELS)))
    for j, c in enumerate(LABELS):
        if c in classes:
            full[:, j] = proba[:, classes.index(c)]
    return np.array(LABELS)[full.argmax(axis=1)], full
