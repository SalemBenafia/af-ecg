"""Single-lead ECG -> explainable features (RR irregularity, P wave, atrial activity, quality)."""
from fractions import Fraction

import neurokit2 as nk
import numpy as np
from scipy import signal, stats

FS = 300            # everything is resampled to the CinC 2017 rate
PRE, POST = 0.25, 0.45   # beat window around each R peak (s)


def to_fs(ecg, fs, target=FS):
    """Resample to `target` Hz (e.g. 250 -> 300 for MIT-BIH AFDB)."""
    if int(round(fs)) == target:
        return np.asarray(ecg, float)
    fr = Fraction(target, int(round(fs))).limit_denominator(1000)
    return signal.resample_poly(np.asarray(ecg, float), fr.numerator, fr.denominator)


def preprocess(ecg, fs):
    """Return (raw at 300 Hz, cleaned, R-peak sample indices)."""
    raw = to_fs(np.nan_to_num(np.asarray(ecg, float)), fs)
    raw = raw - np.median(raw)
    cleaned = nk.ecg_clean(raw, sampling_rate=FS, method="neurokit")
    try:
        _, info = nk.ecg_peaks(cleaned, sampling_rate=FS)
        rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:          # flat or very short signals
        rpeaks = np.array([], dtype=int)
    return raw, cleaned, rpeaks


def rr_features(rpeaks):
    rr_all = np.diff(rpeaks) / FS
    rr = rr_all[(rr_all > 0.25) & (rr_all < 2.5)]
    f = {"n_beats": float(len(rpeaks)),
         "frac_rr_implausible": float(np.mean((rr_all <= 0.25) | (rr_all >= 2.5))) if len(rr_all) else 1.0}
    keys = ["hr_mean", "rr_cv", "rmssd_norm", "pnn50", "rr_mad_norm", "sd1_sd2", "tpr", "rr_entropy", "sampen"]
    if len(rr) < 5:
        return f | {k: np.nan for k in keys}
    d = np.diff(rr)
    m = rr.mean()
    sd1 = np.std(d) / np.sqrt(2)
    sd2 = np.sqrt(max(2 * np.var(rr) - np.var(d) / 2, 1e-12))
    turning = ((rr[1:-1] > rr[:-2]) & (rr[1:-1] > rr[2:])) | ((rr[1:-1] < rr[:-2]) & (rr[1:-1] < rr[2:]))
    hist, _ = np.histogram(rr, bins=16)
    p = hist[hist > 0] / hist.sum()
    try:
        se = float(nk.entropy_sample(rr, delay=1, dimension=2, tolerance=0.2 * rr.std())[0])
    except Exception:
        se = np.nan
    f.update({
        "hr_mean": 60 / m,
        "rr_cv": rr.std(ddof=1) / m,                       # overall variability
        "rmssd_norm": np.sqrt(np.mean(d ** 2)) / m,        # beat-to-beat variability (AF: high)
        "pnn50": float(np.mean(np.abs(d) > 0.05)),
        "rr_mad_norm": np.median(np.abs(rr - np.median(rr))) / np.median(rr),
        "sd1_sd2": sd1 / sd2,                              # Poincare shape (AF: round cloud)
        "tpr": float(turning.mean()),                      # turning-point ratio (random ~ 2/3)
        "rr_entropy": float(-(p * np.log2(p)).sum() / np.log2(16)),
        "sampen": se if np.isfinite(se) else 3.0,          # cap 'no matches' (inf)
    })
    return f


def morphology_features(cleaned, rpeaks):
    a, b = int(PRE * FS), int(POST * FS)
    ok = [r for r in rpeaks if r - a >= 0 and r + b <= len(cleaned)]
    keys = ["p_wave_ratio", "tmpl_corr", "qrs_amp", "fwave_ratio"]
    if len(ok) < 3:
        return {k: np.nan for k in keys}
    beats = np.stack([cleaned[r - a:r + b] for r in ok])
    tmpl = np.median(beats, axis=0)
    qrs = tmpl[a - int(0.05 * FS):a + int(0.05 * FS)]
    qrs_amp = np.ptp(qrs) + 1e-9
    p_seg = signal.detrend(tmpl[a - int(0.25 * FS):a - int(0.08 * FS)])
    corr = [np.corrcoef(bt, tmpl)[0, 1] for bt in beats]
    resid = cleaned.copy()
    for r in ok:                                           # remove QRST -> atrial activity remains
        resid[r - a:r + b] -= tmpl
    fq, pw = signal.welch(resid, FS, nperseg=min(len(resid), 4 * FS))
    band = lambda lo, hi: pw[(fq >= lo) & (fq <= hi)].sum()
    return {
        "p_wave_ratio": np.ptp(p_seg) / qrs_amp,           # AF: no consistent P wave -> small
        "tmpl_corr": float(np.nanmean(corr)),              # beat shape consistency
        "qrs_amp": float(qrs_amp),
        "fwave_ratio": band(4, 9) / (band(1, 40) + 1e-12),  # fibrillatory-wave power
    }


def quality_features(raw, cleaned, rpeaks):
    fq, pw = signal.welch(raw, FS, nperseg=min(len(raw), 4 * FS))
    q = np.nan
    if len(rpeaks) >= 4:
        try:
            q = float(np.nanmean(nk.ecg_quality(cleaned, rpeaks=rpeaks, sampling_rate=FS)))
        except Exception:
            q = np.nan
    lo, hi = np.percentile(cleaned, [1, 99])
    return {
        "duration_s": len(raw) / FS,
        "kurtosis": float(stats.kurtosis(cleaned)),
        "nk_quality": q,
        "hf_noise_ratio": pw[fq > 40].sum() / (pw.sum() + 1e-12),
        "lf_ratio": pw[fq < 0.7].sum() / (pw.sum() + 1e-12),
        "amp_range": float(hi - lo),
    }


def extract_features(ecg, fs):
    raw, cleaned, rpeaks = preprocess(ecg, fs)
    return rr_features(rpeaks) | morphology_features(cleaned, rpeaks) | quality_features(raw, cleaned, rpeaks)


FEATURES = ["n_beats", "frac_rr_implausible", "hr_mean", "rr_cv", "rmssd_norm", "pnn50", "rr_mad_norm",
            "sd1_sd2", "tpr", "rr_entropy", "sampen", "p_wave_ratio", "tmpl_corr", "qrs_amp", "fwave_ratio",
            "duration_s", "kurtosis", "nk_quality", "hf_noise_ratio", "lf_ratio", "amp_range"]
RR_FEATURES = FEATURES[:11]
