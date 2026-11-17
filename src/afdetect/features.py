"""Feature extraction: 21 explainable ECG features (RR, P wave, beat shape, f wave, quality) at 300 Hz.

Pipeline: preprocess (resample, clean, R peaks) -> rr_features + morphology_features + quality_features.
All definitions follow docs/SPEC.md section 6; plain-language explanations are in docs/FEATURES.md."""
from fractions import Fraction

import neurokit2 as nk
import numpy as np
from scipy import signal, stats

FS = 300  # every recording is resampled to the CinC 2017 rate

# Definitions from SPEC section 6 (fixed by the spec, not tuning knobs).
RR_MIN, RR_MAX = 0.25, 2.5   # plausible RR interval range (s), both ends excluded
MIN_RR = 5                   # fewer plausible RR intervals -> RR features are NaN
PRE, POST = 0.25, 0.45       # beat window around each R peak (s)
MIN_BEATS = 3                # fewer complete beats -> no template -> morphology features are NaN
P_WIN = (-0.25, -0.08)       # P-wave window relative to R (s)
QRS_HALF = 0.05              # QRS window is R +- 0.05 s
WELCH_SECONDS = 4            # Welch segment length (s)
SAMPEN_CAP = 3.0             # sample entropy is infinite when no matches are found: cap it

FEATURES = ["n_beats", "frac_rr_implausible", "hr_mean", "rr_cv", "rmssd_norm", "pnn50", "rr_mad_norm",
            "sd1_sd2", "tpr", "rr_entropy", "sampen",
            "p_wave_ratio", "tmpl_corr", "qrs_amp", "fwave_ratio",
            "duration_s", "kurtosis", "nk_quality", "hf_noise_ratio", "lf_ratio", "amp_range"]
RR_FEATURES = FEATURES[:11]
MORPHOLOGY_FEATURES = FEATURES[11:15]
QUALITY_FEATURES = FEATURES[15:]


def to_fs(ecg, fs, target=FS):
    """Resample `ecg` from `fs` Hz to `target` Hz with a polyphase filter (e.g. 250 -> 300 is 6/5)."""
    x = np.asarray(ecg, dtype=np.float64)
    if abs(float(fs) - target) < 1e-6:
        return x.copy()
    ratio = Fraction(target) / Fraction(float(fs)).limit_denominator(1000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator)


def preprocess(ecg, fs):
    """Return (raw, cleaned, rpeaks) at 300 Hz.

    raw: resampled signal minus its median (NaN/inf set to 0).
    cleaned: nk.ecg_clean(method="neurokit").
    rpeaks: R-peak sample indices from nk.ecg_peaks.
    If cleaning or peak detection fails (flat, very short or broken signal), cleaned is a copy of raw
    and rpeaks is an empty int array: never a crash."""
    x = np.nan_to_num(np.asarray(ecg, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x, x.copy(), np.array([], dtype=int)
    raw = to_fs(x, fs)
    raw = raw - np.median(raw)
    try:
        cleaned = np.asarray(nk.ecg_clean(raw, sampling_rate=FS, method="neurokit"), dtype=np.float64)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=FS)
        rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        cleaned, rpeaks = raw.copy(), np.array([], dtype=int)
    return raw, cleaned, rpeaks


def heart_rate(rpeaks, fs=FS):
    """Heart rate in bpm from the median RR interval; NaN if there are fewer than 2 peaks."""
    if len(rpeaks) < 2:
        return float("nan")
    return float(60.0 / np.median(np.diff(rpeaks) / fs))


def _ratio(num, den):
    """num / den as a float, or NaN when the denominator is zero or not finite."""
    return float(num / den) if np.isfinite(den) and den > 0 else float("nan")


def _sample_entropy(rr):
    """Sample entropy of the RR series (m=2, r=0.2*SD); infinite -> SAMPEN_CAP, failure -> NaN."""
    try:
        value = float(nk.entropy_sample(rr, delay=1, dimension=2, tolerance=0.2 * np.std(rr))[0])
    except Exception:
        return float("nan")
    return SAMPEN_CAP if np.isinf(value) else value


def rr_features(rpeaks, fs=FS):
    """The 11 RR-interval features: rhythm speed and (ir)regularity.

    RR intervals outside (0.25, 2.5) s are dropped; with fewer than 5 left, every feature except
    n_beats and frac_rr_implausible is NaN."""
    rr_all = np.diff(np.asarray(rpeaks)) / fs
    plausible = (rr_all > RR_MIN) & (rr_all < RR_MAX)
    out = {"n_beats": float(len(rpeaks)),
           "frac_rr_implausible": float(np.mean(~plausible)) if len(rr_all) else float("nan")}
    rr = rr_all[plausible]
    if len(rr) < MIN_RR:
        return out | {k: float("nan") for k in RR_FEATURES[2:]}
    d = np.diff(rr)
    mean, median = rr.mean(), np.median(rr)
    sd1 = np.std(d) / np.sqrt(2)
    sd2 = np.sqrt(max(2 * np.var(rr) - np.var(d) / 2, 1e-12))
    mid = rr[1:-1]
    turning = ((mid > rr[:-2]) & (mid > rr[2:])) | ((mid < rr[:-2]) & (mid < rr[2:]))
    hist, _ = np.histogram(rr, bins=16)
    p = hist[hist > 0] / hist.sum()
    return out | {
        "hr_mean": 60.0 / mean,
        "rr_cv": float(np.std(rr, ddof=1) / mean),
        "rmssd_norm": float(np.sqrt(np.mean(d ** 2)) / mean),
        "pnn50": float(np.mean(np.abs(d) > 0.05)),
        "rr_mad_norm": float(np.median(np.abs(rr - median)) / median),
        "sd1_sd2": float(sd1 / sd2),
        "tpr": float(turning.mean()),
        "rr_entropy": float(-(p * np.log2(p)).sum() / np.log2(16)),
        "sampen": _sample_entropy(rr),
    }


def _complete_beats(cleaned, rpeaks, fs=FS):
    """R peaks whose full beat window (-PRE, +POST) lies inside the signal, and the window offsets."""
    pre, post = int(round(PRE * fs)), int(round(POST * fs))
    ok = np.array([r for r in rpeaks if r - pre >= 0 and r + post <= len(cleaned)], dtype=int)
    return ok, pre, post


def _beat_model(n, ok, template, pre, post):
    """A signal with the template placed at every beat (later beats overwrite overlaps), zeros elsewhere.

    Subtracting it once removes the QRST without double-subtracting where windows overlap (RR < 0.7 s)."""
    model = np.zeros(n)
    for r in ok:
        model[r - pre:r + post] = template
    return model


def _band_power(freqs, power, lo, hi):
    """Sum of Welch power between lo and hi Hz (both included)."""
    return float(power[(freqs >= lo) & (freqs <= hi)].sum())


def morphology_features(cleaned, rpeaks, fs=FS):
    """P-wave strength, beat-shape consistency, QRS size and f-wave (atrial) power.

    Needs at least 3 complete beats to build the median beat ("template"), otherwise all NaN."""
    ok, pre, post = _complete_beats(cleaned, rpeaks, fs)
    if len(ok) < MIN_BEATS:
        return {k: float("nan") for k in MORPHOLOGY_FEATURES}
    beats = np.stack([cleaned[r - pre:r + post] for r in ok])
    template = np.median(beats, axis=0)
    q = int(round(QRS_HALF * fs))
    qrs_amp = float(np.ptp(template[pre - q:pre + q]))
    p_seg = signal.detrend(template[pre + int(round(P_WIN[0] * fs)):pre + int(round(P_WIN[1] * fs))])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = [np.corrcoef(b, template)[0, 1] if np.std(b) > 0 else np.nan for b in beats]
    residual = cleaned - _beat_model(len(cleaned), ok, template, pre, post)
    freqs, power = signal.welch(residual, fs, nperseg=min(len(residual), WELCH_SECONDS * fs))
    return {
        "p_wave_ratio": _ratio(np.ptp(p_seg), qrs_amp),
        "tmpl_corr": float(np.nanmean(corr)) if np.isfinite(corr).any() else float("nan"),
        "qrs_amp": qrs_amp,
        "fwave_ratio": _ratio(_band_power(freqs, power, 4, 9), _band_power(freqs, power, 1, 40)),
    }


def _ecg_quality(cleaned, rpeaks, fs=FS):
    """Mean NeuroKit beat-quality score; NaN with fewer than 4 peaks or on failure."""
    if len(rpeaks) < 4:
        return float("nan")
    try:
        return float(np.nanmean(nk.ecg_quality(cleaned, rpeaks=rpeaks, sampling_rate=fs)))
    except Exception:
        return float("nan")


def quality_features(raw, cleaned, rpeaks, fs=FS):
    """Signal-quality features: duration, spikiness, NeuroKit quality, high/low-frequency noise, amplitude.

    With less than 1 s of signal only duration_s is filled in."""
    out = {"duration_s": len(raw) / fs}
    if len(raw) < fs:
        return out | {k: float("nan") for k in QUALITY_FEATURES[1:]}
    freqs, power = signal.welch(raw, fs, nperseg=min(len(raw), WELCH_SECONDS * fs))
    total = float(power.sum())
    lo, hi = np.percentile(cleaned, [1, 99])
    with np.errstate(invalid="ignore", divide="ignore"):
        kurt = float(stats.kurtosis(cleaned)) if np.std(cleaned) > 0 else float("nan")
    return out | {
        "kurtosis": kurt,
        "nk_quality": _ecg_quality(cleaned, rpeaks, fs),
        "hf_noise_ratio": _ratio(float(power[freqs > 40].sum()), total),
        "lf_ratio": _ratio(float(power[freqs < 0.7].sum()), total),
        "amp_range": float(hi - lo),
    }


def extract_features(ecg, fs):
    """All 21 features of one recording, as a dict in FEATURES order (NaN where undefined; never raises)."""
    try:
        raw, cleaned, rpeaks = preprocess(ecg, fs)
        f = rr_features(rpeaks) | morphology_features(cleaned, rpeaks) | quality_features(raw, cleaned, rpeaks)
    except Exception:
        f = {}
    return {k: float(f.get(k, np.nan)) for k in FEATURES}
