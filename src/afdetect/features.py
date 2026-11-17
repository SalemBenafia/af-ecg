"""Feature extraction: 21 explainable ECG features (RR, P wave, beat shape, f wave, quality) at 300 Hz.

This file is built in two phases: P3 (resampling, cleaning, R peaks) and P4 (the features)."""
from fractions import Fraction

import neurokit2 as nk
import numpy as np
from scipy import signal

FS = 300  # every recording is resampled to the CinC 2017 rate


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
