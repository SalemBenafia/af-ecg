"""Tests for the synthetic ECG generator (tests/synth.py)."""
import numpy as np
import pytest

from afdetect.features import FS, preprocess
from synth import KINDS, make_ecg


def detected_rr(kind, seed=0):
    """RR intervals (s) found by the real preprocessing on a synthetic ECG."""
    _, _, rpeaks = preprocess(make_ecg(kind, 30, FS, seed=seed), FS)
    return np.diff(rpeaks) / FS


@pytest.mark.parametrize("kind", KINDS)
def test_length_and_determinism(kind):
    x = make_ecg(kind, seconds=12, fs=250, seed=5)
    assert len(x) == 12 * 250
    np.testing.assert_array_equal(x, make_ecg(kind, seconds=12, fs=250, seed=5))
    assert not np.array_equal(x, make_ecg(kind, seconds=12, fs=250, seed=6))


def test_af_rr_more_irregular_than_normal():
    rr_n, rr_a = detected_rr("N"), detected_rr("A")
    assert rr_a.std() / rr_a.mean() > 3 * rr_n.std() / rr_n.mean()


def test_other_rhythm_has_short_and_long_rr():
    rr = detected_rr("O")
    assert np.quantile(rr, 0.1) < 0.6 < 1.0 < np.quantile(rr, 0.9)


def test_noise_has_large_amplitude():
    assert np.ptp(make_ecg("~")) > 5 * np.ptp(make_ecg("N"))


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown kind"):
        make_ecg("X")
