"""MIT-BIH AFDB: 30-s single-rhythm windows to an external features CSV (external test only).

Usage: python -m afdetect.afdb --root data/afdb --out data/features/afdb.csv [--seed 0] [--max-per-label 60]
Labels come from the .atr rhythm annotations: (AFIB -> A, (N -> N, (AFL and (J -> O. Any other rhythm
annotation (e.g. (SVTA) ends the previous rhythm and its windows are not used. AFDB "(N" means "all other
rhythms", not "normal", so the main external numbers are AF sensitivity and specificity.
Never train, tune or choose thresholds on this data.
"""
import argparse
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed

from .data import read_wfdb
from .features import FEATURES, extract_features

RHYTHM = {"(AFIB": "A", "(N": "N", "(AFL": "O", "(J": "O"}
AFDB_LABELS = ["N", "A", "O"]
WINDOW_S = 30


def rhythm_intervals(samples, aux_notes, n_samples):
    """[(start, end, label)] between consecutive rhythm annotations; label None for unmapped rhythms.

    Every aux_note starting with "(" is a rhythm change; the last rhythm runs to the end of the signal."""
    marks = [(int(s), note.strip().rstrip("\x00")) for s, note in zip(samples, aux_notes)
             if note and note.strip().startswith("(")]
    out = []
    for i, (start, note) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else n_samples
        if end > start:
            out.append((start, end, RHYTHM.get(note)))
    return out


def candidate_windows(intervals, window):
    """Non-overlapping windows of `window` samples lying completely inside one labelled rhythm interval."""
    return [(s0, s0 + window, label) for start, end, label in intervals if label is not None
            for s0 in range(start, end - window + 1, window)]


def usable(segment):
    """False for windows with NaN or a flat (constant) signal."""
    return not np.isnan(segment).any() and np.ptp(segment) > 0


def pick_windows(windows, max_per_label, rng):
    """At most max_per_label windows per label, chosen at random, returned in time order."""
    picked = []
    for label in AFDB_LABELS:
        cands = [w for w in windows if w[2] == label]
        if len(cands) > max_per_label:
            cands = [cands[i] for i in sorted(rng.choice(len(cands), max_per_label, replace=False))]
        picked += cands
    return sorted(picked)


def record_rng(seed, name):
    """A random generator fixed by (seed, record name): adding records never changes other records' picks."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def windows_for_record(path, max_per_label=60, seed=0, window_s=WINDOW_S):
    """Feature rows for one AFDB record (channel 0), plus a count of skipped windows."""
    path = Path(path)
    x, fs = read_wfdb(path, channel=0)
    ann = wfdb.rdann(str(path), "atr")
    window = int(round(window_s * fs))
    cands = candidate_windows(rhythm_intervals(ann.sample, ann.aux_note, len(x)), window)
    good = [w for w in cands if usable(x[w[0]:w[1]])]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a, b, label in pick_windows(good, max_per_label, record_rng(seed, path.name)):
            rows.append({"record": f"{path.name}_{a}", "source_record": path.name, "label": label,
                         "start_s": a / fs, "read_ok": 1} | extract_features(x[a:b], fs))
    return rows, len(cands) - len(good)


def records_with_signal(root):
    """AFDB record paths (without extension) that have both a .hea and a .dat file, sorted."""
    return sorted(p.with_suffix("") for p in Path(root).glob("*.hea") if p.with_suffix(".dat").exists())


def main(argv=None):
    """Build the AFDB window features CSV and print windows per record and label."""
    ap = argparse.ArgumentParser(description="AFDB -> 30-s single-rhythm windows -> features CSV.")
    ap.add_argument("--root", default="data/afdb")
    ap.add_argument("--out", default="data/features/afdb.csv")
    ap.add_argument("--max-per-label", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args(argv)

    recs = records_with_signal(a.root)
    skipped_records = sorted({p.stem for p in Path(a.root).glob("*.hea")} - {r.name for r in recs})
    print(f"{len(recs)} records with signal files; without .dat (not used): {skipped_records}")
    parts = Parallel(n_jobs=a.jobs)(delayed(windows_for_record)(r, a.max_per_label, a.seed) for r in recs)
    cols = ["record", "source_record", "label", "start_s", "read_ok"] + FEATURES
    df = pd.DataFrame([row for rows, _ in parts for row in rows], columns=cols)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    table = df.pivot_table(index="source_record", columns="label", values="record", aggfunc="count",
                           fill_value=0).reindex(columns=AFDB_LABELS, fill_value=0)
    print(table.to_string())
    no_windows = sorted({r.name for r in recs} - set(df["source_record"]))
    print(f"Records with a signal but no usable window: {no_windows}")
    print(f"Windows per label: {df['label'].value_counts().reindex(AFDB_LABELS, fill_value=0).to_dict()}; "
          f"skipped (NaN/flat): {sum(n for _, n in parts)}; seed {a.seed}, max {a.max_per_label} per label/record")
    print(f"Saved {out}")
    return df


if __name__ == "__main__":
    main()
