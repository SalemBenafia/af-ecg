"""Tests for afdetect.models on random feature tables with a planted signal."""
import numpy as np
import pandas as pd
import pytest

from afdetect.data import LABELS
from afdetect.features import FEATURES
from afdetect.models import RRRule, make_model, predict


def feature_table(n_per_class=80, seed=0):
    """Random features where rmssd_norm is high for A, p_wave_ratio high for O and amp_range high for ~."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(LABELS, n_per_class)
    X = pd.DataFrame(rng.normal(size=(len(labels), len(FEATURES))), columns=FEATURES)
    X["rmssd_norm"] = np.abs(X["rmssd_norm"]) * 0.05 + np.where(labels == "A", 0.4, 0.0)
    X["p_wave_ratio"] += np.where(labels == "O", 3.0, 0.0)
    X["amp_range"] += np.where(labels == "~", 3.0, 0.0)
    X.loc[0, "rmssd_norm"] = np.nan  # missing values must be handled
    return X, labels


def test_rr_rule_threshold_and_two_columns():
    X, y = feature_table()
    rule = RRRule().fit(X, y)
    assert 0.2 < rule.threshold_ < 0.4  # between the Normal-like values and the AF values
    proba = rule.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_array_equal(proba.sum(axis=1), 1)
    assert rule.train_af_f1_ > 0.95


def test_predict_maps_to_labels_order_with_zeros():
    X, y = feature_table()
    model, cols = make_model("rr_rule")
    model.fit(X[cols], y)
    labels, proba = predict(model, cols, X)
    assert proba.shape == (len(X), 4)
    np.testing.assert_array_equal(proba[:, 2:], 0)  # rr_rule has no O or ~ column
    assert set(labels) <= {"N", "A"}
    assert (labels[y == "A"] == "A").mean() > 0.95


@pytest.mark.parametrize("name", ["logreg_rr", "gbm"])
def test_learned_models_beat_chance(name):
    X, y = feature_table()
    model, cols = make_model(name, seed=0)
    model.fit(X[cols], y)
    labels, proba = predict(model, cols, X)
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-6)
    assert (labels == y).mean() > 0.4  # chance is 0.25
    assert (labels[y == "A"] == "A").mean() > 0.9  # the planted AF signal is found


def test_gbm_settings_follow_spec():
    model, cols = make_model("gbm", seed=7)
    p = model.get_params()
    assert (p["max_iter"], p["learning_rate"], p["max_leaf_nodes"], p["l2_regularization"]) == (400, 0.05, 31, 1.0)
    assert p["class_weight"] == "balanced" and p["early_stopping"] is True and p["random_state"] == 7
    assert cols == FEATURES


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("cnn")
