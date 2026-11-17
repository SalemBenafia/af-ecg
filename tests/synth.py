"""Synthetic single-lead ECGs for tests: N (normal), A (AF), O (ectopic beats) and ~ (noise).

Each beat is a sum of Gaussian P, Q, R, S and T waves placed around the R time. Signals are in mV.
These are test fixtures, not physiology: they only need the features each class is known for."""
import numpy as np

# (offset from R in s, amplitude in mV, width sigma in s)
P_WAVE = (-0.16, 0.15, 0.025)
NORMAL_QRST = [(-0.03, -0.12, 0.010), (0.0, 1.0, 0.012), (0.03, -0.25, 0.012), (0.25, 0.30, 0.050)]
WIDE_QRST = [(-0.05, -0.15, 0.030), (0.0, 1.2, 0.036), (0.07, -0.50, 0.036), (0.30, -0.35, 0.060)]
KINDS = ("N", "A", "O", "~")


def _add_wave(x, t, centre, amp, sigma):
    """Add one Gaussian wave to x (in place), only where it is not negligible."""
    near = np.abs(t - centre) < 5 * sigma
    x[near] += amp * np.exp(-0.5 * ((t[near] - centre) / sigma) ** 2)


def _add_beat(x, t, r_time, waves, with_p):
    """Add one beat (optional P wave + the given QRS/T waves) with its R peak at r_time."""
    for offset, amp, sigma in ([P_WAVE] if with_p else []) + waves:
        _add_wave(x, t, r_time + offset, amp, sigma)


def _rr_series(kind, seconds, rng):
    """RR intervals (s) and, per beat, whether it is a premature wide beat."""
    rr, wide = [], []
    every = int(rng.choice([2, 3]))  # O: every 2nd or 3rd beat is premature
    while sum(rr) < seconds + 2:
        i = len(rr)
        if kind == "N":
            rr.append(0.8 * (1 + 0.03 * rng.standard_normal()) * (1 + 0.04 * np.sin(2 * np.pi * i / 5)))
            wide.append(False)
        elif kind == "A":
            rr.append(0.7 * rng.uniform(0.6, 1.4))
            wide.append(False)
        elif kind == "O":
            premature = i % every == every - 1
            after_premature = i % every == 0 and i > 0
            rr.append(0.8 * (0.6 if premature else 1.4 if after_premature else 1.0)
                      * (1 + 0.02 * rng.standard_normal()))
            wide.append(premature)
        else:  # "~": a normal-looking rhythm underneath the noise
            rr.append(0.8 * (1 + 0.05 * rng.standard_normal()))
            wide.append(False)
    return np.array(rr), np.array(wide)


def make_ecg(kind, seconds=30, fs=300, seed=0):
    """Synthetic ECG of the given kind ("N", "A", "O" or "~"), `seconds` long at `fs` Hz, in mV."""
    if kind not in KINDS:
        raise ValueError(f"Unknown kind {kind!r}: expected one of {KINDS}")
    rng = np.random.default_rng(seed)
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    x = np.zeros(n)
    rr, wide = _rr_series(kind, seconds, rng)
    r_times = 0.3 + np.cumsum(rr) - rr[0]  # first beat at 0.3 s
    for r_time, is_wide in zip(r_times, wide):
        if r_time > seconds + 0.5:
            break
        _add_beat(x, t, r_time, WIDE_QRST if is_wide else NORMAL_QRST, with_p=(kind != "A" and not is_wide))
    if kind == "A":  # fibrillatory waves: 5-7 Hz, slowly changing amplitude
        f_hz = rng.uniform(5, 7)
        amp = rng.uniform(0.05, 0.08) * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t))
        x += amp * np.sin(2 * np.pi * f_hz * t + rng.uniform(0, 2 * np.pi))
    if kind == "~":
        x = 0.3 * x + 0.4 * rng.standard_normal(n)
        x += rng.uniform(0.7, 1.3) * np.sin(2 * np.pi * rng.uniform(0.2, 0.5) * t)  # baseline wander
        for _ in range(int(rng.integers(3, 6))):  # motion bursts
            start = rng.uniform(0, max(seconds - 1.5, 0.1))
            burst = (t >= start) & (t < start + rng.uniform(0.5, 1.5))
            x[burst] += rng.uniform(2, 4) * rng.standard_normal(burst.sum())
    return x + 0.01 * rng.standard_normal(n)
