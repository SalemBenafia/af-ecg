"""External test set: MIT-BIH Atrial Fibrillation Database -> 30-s windows -> features CSV.
    python -m afdetect.afdb --root data/afdb --out data/features/afdb.csv
Rhythm labels come from the .atr annotations: (AFIB -> A, (N -> N, (AFL / (J -> O.
Only windows that lie completely inside one rhythm are used. Lead: first ECG channel.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed

from .features import extract_features

RHYTHM = {"(AFIB": "A", "(N": "N", "(AFL": "O", "(J": "O"}
def rhythm_intervals(ann, n_samples):
    """[(start, end, label)] from rhythm-change annotations."""
    marks = [(s, RHYTHM.get(a.strip())) for s, a in zip(ann.sample, ann.aux_note) if a.strip() in RHYTHM]
    out = []
    for i, (s, lab) in enumerate(marks):
        e = marks[i + 1][0] if i + 1 < len(marks) else n_samples
        out.append((int(s), int(e), lab))
    return out


def windows_for_record(path, win_s=30, max_per_label=60, seed=0):
    rec = wfdb.rdrecord(str(path), channels=[0])
    fs, x = float(rec.fs), rec.p_signal[:, 0]
    ann = wfdb.rdann(str(path), "atr")
    rng = np.random.default_rng(seed)
    w = int(win_s * fs)
    picked = []
    for lab in ("N", "A", "O"):
        cands = [(s0, s0 + w) for s, e, lb in rhythm_intervals(ann, len(x)) if lb == lab
                 for s0 in range(s, e - w + 1, w)]
        if len(cands) > max_per_label:
            cands = [cands[i] for i in rng.choice(len(cands), max_per_label, replace=False)]
        picked += [(a, b, lab) for a, b in cands]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a, b, lab in picked:
            seg = x[a:b]
            if np.isnan(seg).any() or np.ptp(seg) == 0:     # skip gaps / flat blocks
                continue
            rows.append({"record": f"{Path(path).name}_{a}", "source_record": Path(path).name,
                         "label": lab} | extract_features(seg, fs))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_per_label", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    recs = sorted(p.with_suffix("") for p in Path(a.root).glob("*.hea") if p.with_suffix(".dat").exists())
    print(f"{len(recs)} records with signals")
    parts = Parallel(n_jobs=a.jobs, verbose=5)(delayed(windows_for_record)(r, 30, a.max_per_label) for r in recs)
    df = pd.DataFrame([row for p in parts for row in p])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby(["label"]).size().to_string())


if __name__ == "__main__":
    main()
