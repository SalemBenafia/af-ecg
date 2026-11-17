"""Visual check of R-peak detection on real CinC 2017 recordings (2 random per class).

Usage: python scripts/check_peaks.py --root data/training2017 --out results/explore --seed 1 [--split train]
Records are drawn from TRAIN by default: never pick detector fixes by looking at test recordings.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from afdetect.data import LABELS, NAMES, load_reference, read_wfdb  # noqa: E402
from afdetect.features import FS, heart_rate, preprocess  # noqa: E402

PLOT_SECONDS = 10
PER_CLASS = 2


def restrict_to_split(ref, splits_csv, split):
    """Keep only records of one split (default TRAIN), so peak-detection fixes are never chosen on test data."""
    if split == "all":
        return ref
    splits = pd.read_csv(splits_csv, dtype=str)
    keep = set(splits.loc[splits["split"] == split, "record"])
    return ref[ref["record"].isin(keep)]


def pick_records(ref, seed, per_class=PER_CLASS):
    """Randomly pick per_class records for every label (fixed seed)."""
    rng = np.random.default_rng(seed)
    return {c: list(rng.choice(ref.loc[ref["label"] == c, "record"].to_numpy(), per_class, replace=False))
            for c in LABELS}


def peak_summary(rpeaks, n_samples):
    """Number of peaks, heart rate and the fraction of implausible RR intervals (outside 0.25-2.5 s)."""
    rr = np.diff(rpeaks) / FS
    implausible = float(np.mean((rr <= 0.25) | (rr >= 2.5))) if len(rr) else float("nan")
    return {"n_peaks": int(len(rpeaks)), "seconds": n_samples / FS, "hr": heart_rate(rpeaks),
            "frac_rr_implausible": implausible}


def plot_peaks(root, picks, out_png, seconds=PLOT_SECONDS):
    """Grid of cleaned ECGs (first `seconds`) with detected R peaks; returns the per-record summaries."""
    n_cols = max(len(v) for v in picks.values())
    fig, axes = plt.subplots(len(LABELS), n_cols, figsize=(7 * n_cols, 2.4 * len(LABELS)), squeeze=False)
    summaries = {}
    for row, label in enumerate(LABELS):
        for col, rec in enumerate(picks[label]):
            x, fs = read_wfdb(Path(root) / rec)
            _, cleaned, rpeaks = preprocess(x, fs)
            s = summaries[rec] = {"label": label, **peak_summary(rpeaks, len(cleaned))}
            n = min(len(cleaned), seconds * FS)
            shown = rpeaks[rpeaks < n]
            ax = axes[row, col]
            ax.plot(np.arange(n) / FS, cleaned[:n], lw=0.7, color="#1f4e79")
            ax.plot(shown / FS, cleaned[shown], "o", ms=4, color="#d62728")
            ax.set_title(f"{label} · {NAMES[label]} · {rec} · {s['n_peaks']} peaks in {s['seconds']:.0f} s · "
                         f"HR {s['hr']:.0f} bpm", fontsize=9, loc="left")
            ax.set_xlim(0, seconds)
            ax.set_ylabel("mV (cleaned)")
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return summaries


def main(argv=None):
    """Pick records, plot peaks.png and print a per-record summary."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/training2017")
    ap.add_argument("--out", default="results/explore")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--splits", default="configs/splits.csv")
    ap.add_argument("--split", default="train", choices=["train", "val", "all"],
                    help="sample only from this split (test is deliberately not offered)")
    a = ap.parse_args(argv)

    picks = pick_records(restrict_to_split(load_reference(a.root), a.splits, a.split), a.seed)
    out_png = Path(a.out) / "peaks.png"
    summaries = plot_peaks(a.root, picks, out_png)
    for rec, s in summaries.items():
        print(f"{rec} {s['label']}: {s['n_peaks']:3d} peaks / {s['seconds']:4.1f} s, HR {s['hr']:5.1f} bpm, "
              f"implausible RR {s['frac_rr_implausible']:.2f}")
    print(f"Saved {out_png} (seed {a.seed})")
    return summaries


if __name__ == "__main__":
    main()
