"""Tests for afdetect.metrics with hand-computed values."""
import numpy as np
import pytest

from afdetect.metrics import bootstrap_ci, confusion, coverage_curve, summarize

# index:     0    1    2    3    4    5    6    7    8    9
Y_TRUE = ["N", "N", "N", "N", "A", "A", "A", "O", "O", "~"]
Y_PRED = ["N", "N", "N", "A", "A", "A", "O", "O", "N", "~"]
P_AF = [0.1, 0.2, 0.1, 0.6, 0.9, 0.8, 0.4, 0.3, 0.2, 0.1]


def test_summarize_hand_computed():
    s = summarize(Y_TRUE, Y_PRED, P_AF)
    # N: TP 3 (0-2), FP 1 (8: O->N), FN 1 (3: N->A)  -> F1 = 2*3 / (2*3 + 1 + 1) = 0.75
    assert s["F1_N"] == pytest.approx(0.75)
    # A: TP 2 (4, 5), FP 1 (3), FN 1 (6: A->O)        -> F1 = 4 / 6
    assert s["F1_A"] == pytest.approx(4 / 6)
    # O: TP 1 (7), FP 1 (6), FN 1 (8)                 -> F1 = 2 / 4
    assert s["F1_O"] == pytest.approx(0.5)
    # ~: TP 1, no errors                              -> F1 = 1
    assert s["F1_~"] == pytest.approx(1.0)
    assert s["challenge_F1"] == pytest.approx((0.75 + 4 / 6 + 0.5) / 3)  # ~ is excluded
    assert s["accuracy"] == pytest.approx(7 / 10)  # correct: 0, 1, 2, 4, 5, 7, 9
    # AF as positive: TP 2, FN 1, FP 1, TN 6
    assert s["AF_sensitivity"] == pytest.approx(2 / 3)
    assert s["AF_specificity"] == pytest.approx(6 / 7)
    assert s["AF_PPV"] == pytest.approx(2 / 3)
    # AUROC: 3 AF x 7 non-AF = 21 pairs; 0.9 and 0.8 beat all 7, 0.4 beats 6 (not 0.6) -> 20 / 21
    assert s["AF_AUROC"] == pytest.approx(20 / 21)
    assert s["n"] == 10


def test_summarize_undefined_values_are_nan():
    s = summarize(["N", "N"], ["N", "N"], [0.1, 0.2])
    assert np.isnan(s["AF_sensitivity"]) and np.isnan(s["AF_PPV"]) and np.isnan(s["AF_AUROC"])
    assert s["AF_specificity"] == 1.0
    assert np.isnan(summarize(Y_TRUE, Y_PRED)["AF_AUROC"])  # no p_af given


def test_confusion_hand_computed():
    expected = [[3, 1, 0, 0],   # true N
                [0, 2, 1, 0],   # true A
                [1, 0, 1, 0],   # true O
                [0, 0, 0, 1]]   # true ~
    np.testing.assert_array_equal(confusion(Y_TRUE, Y_PRED), expected)


def noisy_predictions(n=300, error_rate=0.25, seed=0):
    """Random labels and predictions that are right about 75% of the time."""
    rng = np.random.default_rng(seed)
    y_true = rng.choice(["N", "A", "O", "~"], size=n, p=[0.55, 0.15, 0.25, 0.05])
    y_pred = np.where(rng.random(n) < error_rate, rng.choice(["N", "A", "O", "~"], size=n), y_true)
    return y_true, y_pred


@pytest.mark.parametrize("metric", ["challenge_F1", "AF_sensitivity"])
def test_bootstrap_ci_contains_point_estimate_and_is_reproducible(metric):
    y_true, y_pred = noisy_predictions()
    point = summarize(y_true, y_pred)[metric]
    lo, hi = bootstrap_ci(y_true, y_pred, metric, n_boot=500, seed=1)
    assert lo < point < hi
    assert (lo, hi) == bootstrap_ci(y_true, y_pred, metric, n_boot=500, seed=1)


def test_bootstrap_by_groups():
    y_true, y_pred = noisy_predictions(60)
    point = summarize(y_true, y_pred)["challenge_F1"]
    # One group: every resample is the full data set -> zero-width CI at the point estimate.
    lo, hi = bootstrap_ci(y_true, y_pred, groups=np.zeros(60), n_boot=50)
    assert lo == pytest.approx(point) and hi == pytest.approx(point)
    # Six groups of 10 records: a proper interval.
    lo, hi = bootstrap_ci(y_true, y_pred, groups=np.repeat(np.arange(6), 10), n_boot=200)
    assert lo <= point <= hi and lo < hi


def test_bootstrap_auroc_uses_p_af():
    lo, hi = bootstrap_ci(Y_TRUE * 5, Y_PRED * 5, "AF_AUROC", n_boot=100, p_af=P_AF * 5)
    assert 0.8 < lo <= hi <= 1.0


def test_coverage_curve():
    y_true = ["N"] * 20
    y_pred = ["N"] * 20
    confidence = np.r_[np.full(10, 0.4), np.full(10, 0.9)]
    rows = coverage_curve(y_true, y_pred, confidence, [0.3, 0.5, 0.95])
    assert [r["n_kept"] for r in rows] == [20, 10, 0]
    assert [r["kept_pct"] for r in rows] == [100.0, 50.0, 0.0]
    assert rows[0]["accuracy"] == 1.0
    assert np.isnan(rows[1]["accuracy"]) and np.isnan(rows[2]["challenge_F1"])  # <= 10 kept -> NaN
