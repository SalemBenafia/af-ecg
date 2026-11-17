"""Tests for afdetect.features: resampling, preprocessing, heart rate and the 21 features."""
import numpy as np
import pandas as pd
import pytest

from afdetect import features as F
from afdetect.features import (FEATURES, FS, RR_FEATURES, extract_features, heart_rate, morphology_features,
                               preprocess, quality_features, rr_features, to_fs)
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


# ---------- feature lists ----------

def test_feature_lists():
    assert len(FEATURES) == 21 and len(set(FEATURES)) == 21
    assert RR_FEATURES == FEATURES[:11]
    assert F.MORPHOLOGY_FEATURES + F.QUALITY_FEATURES == FEATURES[11:]


# ---------- RR features ----------

def test_rr_features_regular_rhythm():
    f = rr_features(np.arange(0, 30 * FS, FS))  # one beat per second
    assert list(f) == RR_FEATURES
    assert f["n_beats"] == 30 and f["frac_rr_implausible"] == 0
    assert f["hr_mean"] == pytest.approx(60)
    assert f["rr_cv"] == f["rmssd_norm"] == f["pnn50"] == f["rr_mad_norm"] == pytest.approx(0)
    assert np.isfinite(f["sampen"])


def test_rr_features_too_few_intervals_gives_nan():
    f = rr_features(np.array([0, 300, 600, 900]))  # 3 RR intervals < 5
    assert f["n_beats"] == 4 and f["frac_rr_implausible"] == 0
    assert all(np.isnan(f[k]) for k in RR_FEATURES[2:])
    for peaks in (np.array([], dtype=int), np.array([100])):
        g = rr_features(peaks)
        assert g["n_beats"] == len(peaks) and np.isnan(g["frac_rr_implausible"])


def test_rr_features_drop_implausible_intervals():
    peaks = np.r_[np.arange(0, 10 * FS, FS), 10 * FS + 3 * FS + np.arange(0, 6 * FS, FS)]  # one 4-s gap
    f = rr_features(peaks)
    assert f["frac_rr_implausible"] == pytest.approx(1 / (len(peaks) - 1))
    assert f["hr_mean"] == pytest.approx(60)  # the 4-s gap is not used


def test_sampen_infinite_is_capped(monkeypatch):
    monkeypatch.setattr(F.nk, "entropy_sample", lambda *a, **k: (np.inf, {}))
    rng = np.random.default_rng(0)
    peaks = np.cumsum(rng.integers(200, 300, 20))
    assert rr_features(peaks)["sampen"] == F.SAMPEN_CAP


# ---------- morphology and quality ----------

def test_morphology_needs_three_complete_beats():
    _, cleaned, _ = preprocess(make_ecg("N", 10), FS)
    f = morphology_features(cleaned, np.array([10, 1000, 2000]))  # the first beat is incomplete
    assert all(np.isnan(v) for v in f.values())


def test_fwave_low_when_beat_windows_overlap():
    """At 120 bpm (RR 0.5 s < 0.7 s window) the template is still removed once, not twice.

    Deterministic signal: ours gives about 0.12; subtracting the template at every beat (double
    subtraction where windows overlap, as in the reference code) gives about 0.17 and fails."""
    t = np.arange(30 * FS) / FS
    x = np.zeros_like(t)
    for r in np.arange(0.5, 29.5, 0.5):
        x += np.exp(-0.5 * ((t - r) / 0.012) ** 2) + 0.3 * np.exp(-0.5 * ((t - r - 0.2) / 0.04) ** 2)
    _, cleaned, rpeaks = preprocess(x, FS)
    f = morphology_features(cleaned, rpeaks)
    assert f["tmpl_corr"] > 0.95
    assert f["fwave_ratio"] < 0.15


def test_quality_features_short_signal():
    f = quality_features(np.zeros(100), np.zeros(100), np.array([], dtype=int))
    assert f["duration_s"] == pytest.approx(100 / FS)
    assert all(np.isnan(f[k]) for k in F.QUALITY_FEATURES[1:])


@pytest.mark.parametrize("ecg", [np.zeros(30 * FS), np.zeros(30), np.array([])], ids=["flat", "0.1s", "empty"])
def test_extract_features_never_crashes(ecg):
    f = extract_features(ecg, FS)
    assert list(f) == FEATURES
    assert all(isinstance(v, float) for v in f.values())
    assert f["n_beats"] == 0


# ---------- class behaviour on synthetic ECGs (8 per class, median) ----------

@pytest.fixture(scope="module")
def class_medians():
    rows = [{"kind": k, **extract_features(make_ecg(k, 30, FS, seed=s), FS)} for k in "NAO~" for s in range(8)]
    return pd.DataFrame(rows).groupby("kind")[FEATURES].median()


def test_af_more_irregular_than_normal(class_medians):
    m = class_medians
    assert m.loc["A", "rmssd_norm"] > 3 * m.loc["N", "rmssd_norm"]


def test_af_has_weaker_p_wave_than_normal(class_medians):
    m = class_medians
    assert m.loc["A", "p_wave_ratio"] < 0.5 * m.loc["N", "p_wave_ratio"]


def test_af_has_more_f_wave_power(class_medians):
    m = class_medians
    assert m.loc["A", "fwave_ratio"] > m.loc["N", "fwave_ratio"]


def test_other_rhythm_lower_rr_entropy_than_af(class_medians):
    m = class_medians
    assert m.loc["O", "rr_entropy"] < m.loc["A", "rr_entropy"]


def test_noise_has_inconsistent_beats_and_hf_noise(class_medians):
    m = class_medians
    assert m.loc["~", "tmpl_corr"] < m.loc["N", "tmpl_corr"] - 0.3
    assert m.loc["~", "hf_noise_ratio"] > 10 * m.loc["N", "hf_noise_ratio"]
