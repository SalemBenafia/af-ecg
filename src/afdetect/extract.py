"""CLI: extract features for every CinC 2017 record into one CSV.

Usage: python -m afdetect.extract --root data/training2017 --out data/features/cinc2017.csv [--jobs -1] [--limit N]
Every record in REFERENCE.csv gets exactly one row: a recording that cannot be read or processed gets
NaN features (and read_ok = 0), never a missing row.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import LABELS, load_reference, read_wfdb
from .features import FEATURES, extract_features


def features_for_record(root, record):
    """Features of one WFDB record as a dict (record, read_ok, 21 features); NaN features if it fails."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ecg, fs = read_wfdb(Path(root) / record)
        except Exception:
            return {"record": record, "read_ok": 0} | {k: np.nan for k in FEATURES}
        return {"record": record, "read_ok": 1} | extract_features(ecg, fs)


def extract_all(root, ref, jobs=-1):
    """Features for every record of `ref`, merged with its labels, in the order of `ref`."""
    rows = Parallel(n_jobs=jobs)(delayed(features_for_record)(root, r) for r in ref["record"])
    feats = pd.DataFrame(rows, columns=["record", "read_ok"] + FEATURES)
    out = ref[["record", "label"]].merge(feats, on="record", how="left", validate="one_to_one")
    assert len(out) == len(ref), "a record went missing during extraction"
    return out


def report(df):
    """Text summary: class counts, failed reads and NaN count per feature."""
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    nan = df[FEATURES].isna().sum()
    lines = [f"Records: {len(df)}", "Class counts: " + ", ".join(f"{c} {n}" for c, n in counts.items()),
             f"Unreadable records: {int((df['read_ok'] == 0).sum())}",
             f"Rows with all features NaN: {int(df[FEATURES].isna().all(axis=1).sum())}",
             "NaN values per feature:"]
    lines += [f"  {k:<20} {int(v)}" for k, v in nan.items()]
    return "\n".join(lines)


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Extract the 21 features for every CinC 2017 record.")
    ap.add_argument("--root", default="data/training2017")
    ap.add_argument("--out", default="data/features/cinc2017.csv")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N records (for a quick test)")
    a = ap.parse_args(argv)

    ref = load_reference(a.root)
    if a.limit is not None:
        ref = ref.head(a.limit)
    df = extract_all(a.root, ref, a.jobs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(report(df))
    print(f"Saved {out}")
    return df


if __name__ == "__main__":
    main()
