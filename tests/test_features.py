"""Tests for afdetect.features (P3 part: resampling, preprocessing, heart rate)."""
import numpy as np
import pytest

from afdetect.features import FS, heart_rate, preprocess, to_fs
from synth import make_ecg


def test_to_fs_lengths():
    assert len(to_fs(np.zeros(2500), 250)) == 3000  # 250 -> 300 Hz is 6/5
    assert len(to_fs(np.zeros(5000), 500)) == 3000
    x = np.arange(10.0)
    np.testing.assert_array_equal(to_fs(x, 300), x)


def test_to_fs_keeps_frequency():
    fs = 250
    t = np.arange(10 * fs) / fs
    y = to_fs(np.sin(2 * np.pi * 5 * t), fs)
    freqs = np.fft.rfftfreq(len(y), 1 / FS)
    assert freqs[np.argmax(np.abs(np.fft.rfft(y)))] == pytest.approx(5, abs=0.1)


@pytest.mark.parametrize("fs", [300, 250])
def test_normal_ecg_heart_rate_and_beats(fs):
    seconds = 30
    raw, cleaned, rpeaks = preprocess(make_ecg("N", seconds, fs, seed=3), fs)
    assert len(raw) == len(cleaned) == seconds * FS
    assert abs(np.median(raw)) < 1e-9
    assert 50 <= heart_rate(rpeaks) <= 100
    assert abs(len(rpeaks) - seconds / 0.8) <= 2  # synthetic N has a mean RR of 0.8 s
    assert rpeaks.dtype.kind == "i"


@pytest.mark.parametrize("ecg", [np.zeros(30 * FS), np.zeros(30), np.array([]),
                                 np.r_[np.full(50, np.nan), make_ecg("N", 10)]],
                         ids=["flat", "0.1s", "empty", "with-nan"])
def test_preprocess_never_crashes(ecg):
    raw, cleaned, rpeaks = preprocess(ecg, FS)
    assert len(raw) == len(cleaned) == len(ecg)
    assert rpeaks.dtype.kind == "i"
    assert np.all((rpeaks >= 0) & (rpeaks < max(len(raw), 1)))


def test_flat_signal_has_no_beats_and_nan_heart_rate():
    _, _, rpeaks = preprocess(np.zeros(30 * FS), FS)
    assert len(rpeaks) == 0
    assert np.isnan(heart_rate(rpeaks))


def test_heart_rate_from_regular_peaks():
    assert heart_rate(np.arange(0, 3000, 300)) == pytest.approx(60.0)  # one beat per second
    assert np.isnan(heart_rate(np.array([10])))
