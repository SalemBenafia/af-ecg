"""First look at CinC 2017: class counts, recording lengths and example ECGs per class.

Usage: python scripts/explore.py --root data/training2017 --out results/explore --seed 0
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import wfdb  # noqa: E402

from afdetect.data import LABELS, NAMES, load_reference, read_wfdb  # noqa: E402

PLOT_SECONDS = 10
PER_CLASS = 2


def class_counts(ref):
    """Number and percentage of recordings per label, in LABELS order."""
    counts = ref["label"].value_counts().reindex(LABELS, fill_value=0)
    return pd.DataFrame({"name": [NAMES[c] for c in LABELS], "count": counts.to_numpy(),
                         "percent": (100 * counts / counts.sum()).round(1).to_numpy()}, index=LABELS)


def recording_lengths(root, ref):
    """Recording length in seconds for every record, read from the .hea headers only (fast)."""
    lengths = []
    for rec in ref["record"]:
        h = wfdb.rdheader(str(Path(root) / rec))
        lengths.append(h.sig_len / h.fs)
    return ref.assign(seconds=lengths)


def length_bins(seconds):
    """Counts of recordings shorter than, exactly at, and longer than 30 s."""
    s = np.asarray(seconds)
    exact = np.isclose(s, 30.0)
    return {"< 30 s": int(((s < 30) & ~exact).sum()), "= 30 s": int(exact.sum()),
            "> 30 s": int(((s > 30) & ~exact).sum())}


def pick_examples(ref, seed, per_class=PER_CLASS):
    """Randomly pick per_class records for every label (fixed seed)."""
    rng = np.random.default_rng(seed)
    return {c: list(rng.choice(ref.loc[ref["label"] == c, "record"].to_numpy(), per_class, replace=False))
            for c in LABELS}


def plot_examples(root, picks, out_png, seconds=PLOT_SECONDS):
    """One row per class, one column per picked record: the first `seconds` of the raw ECG in mV."""
    n_cols = max(len(v) for v in picks.values())
    fig, axes = plt.subplots(len(LABELS), n_cols, figsize=(7 * n_cols, 2.4 * len(LABELS)), squeeze=False)
    for row, label in enumerate(LABELS):
        for col, rec in enumerate(picks[label]):
            x, fs = read_wfdb(Path(root) / rec)
            n = min(len(x), int(seconds * fs))
            ax = axes[row, col]
            ax.plot(np.arange(n) / fs, x[:n], lw=0.7, color="#1f4e79")
            ax.set_title(f"{label} · {NAMES[label]} · {rec} ({len(x) / fs:.0f} s)", fontsize=10, loc="left")
            ax.set_xlim(0, seconds)
            ax.set_ylabel("mV")
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Print counts and length statistics, and save examples.png."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/training2017")
    ap.add_argument("--out", default="results/explore")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    ref = load_reference(a.root)
    print(f"Recordings: {len(ref)}\n\nClass counts:")
    print(class_counts(ref).to_string())

    lens = recording_lengths(a.root, ref)
    print("\nRecording length (s):")
    print(lens["seconds"].describe().round(1).to_string())
    print("\nLength bins:", length_bins(lens["seconds"]))
    print("\nMedian length per class (s):")
    print(lens.groupby("label")["seconds"].median().reindex(LABELS).round(1).to_string())

    picks = pick_examples(ref, a.seed)
    out_png = Path(a.out) / "examples.png"
    plot_examples(a.root, picks, out_png)
    print(f"\nExamples (seed {a.seed}): {picks}\nSaved {out_png}")


if __name__ == "__main__":
    main()
