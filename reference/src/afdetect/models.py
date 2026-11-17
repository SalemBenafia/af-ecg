"""Three models, from simplest to strongest. All share fit(X, y) / predict_proba(X) / classes_."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import LABELS
from .features import FEATURES, RR_FEATURES


class RRRule:
    """Baseline 1: 'AF if beat-to-beat variability (normalised RMSSD) > threshold', else Normal.
    The threshold is the one that maximises AF F1 on the training data."""
    classes_ = np.array(LABELS)
    feature = "rmssd_norm"

    def fit(self, X, y):
        x = X[self.feature].fillna(0).to_numpy()
        is_af = np.asarray(y) == "A"
        best = (-1, 0.1)
        for th in np.quantile(x, np.linspace(0.5, 0.99, 100)):
            p = x > th
            tp, fp, fn = (p & is_af).sum(), (p & ~is_af).sum(), (~p & is_af).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
            best = max(best, (f1, th))
        self.threshold_ = float(best[1])
        return self

    def predict_proba(self, X):
        af = (X[self.feature].fillna(0).to_numpy() > self.threshold_).astype(float)
        proba = np.zeros((len(X), 4))
        proba[:, 0], proba[:, 1] = 1 - af, af
        return proba


def make_model(name, seed=0):
    if name == "rr_rule":
        return RRRule(), ["rmssd_norm"]
    if name == "logreg_rr":       # Baseline 2: linear model on RR features only
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=3000, class_weight="balanced"))
        return m, RR_FEATURES
    if name == "gbm":             # Main model: gradient boosting on all features (handles NaN itself)
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, class_weight="balanced",
                                           early_stopping=True, validation_fraction=0.15,
                                           random_state=seed)
        return m, FEATURES
    raise ValueError(name)


def predict(model, cols, X):
    """Return (predicted labels, probability matrix in LABELS order)."""
    proba = model.predict_proba(X[cols])
    order = [list(model.classes_).index(c) if c in model.classes_ else None for c in LABELS]
    full = np.column_stack([proba[:, i] if i is not None else np.zeros(len(X)) for i in order])
    return np.array(LABELS)[full.argmax(1)], full
