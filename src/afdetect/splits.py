"""Stratified 70/15/15 train/val/test split, written once to configs/splits.csv.

Usage: python -m afdetect.splits --root data/training2017 --out configs/splits.csv
The CLI refuses to overwrite an existing split: the split is made once and never regenerated.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import LABELS, load_reference

SPLITS = ["train", "val", "test"]


def make_splits(ref, seed=42, val=0.15, test=0.15):
    """Assign every record to train, val or test, stratified by label.

    First the test part is split off, then val is split off the rest, so that the final
    fractions are about (1 - val - test) / val / test. Returns columns record, label, split
    in the original order of `ref`."""
    rest, test_part = train_test_split(ref, test_size=test, stratify=ref["label"], random_state=seed)
    train_part, val_part = train_test_split(rest, test_size=val / (1 - test), stratify=rest["label"],
                                            random_state=seed)
    split = pd.Series("train", index=ref.index)
    split[val_part.index] = "val"
    split[test_part.index] = "test"
    return ref[["record", "label"]].assign(split=split)


def split_table(splits):
    """Split x label count table with totals, in a fixed row and column order."""
    table = pd.crosstab(splits["split"], splits["label"], margins=True, margins_name="total")
    return table.reindex(index=SPLITS + ["total"], columns=LABELS + ["total"], fill_value=0)


def main(argv=None):
    """CLI: write the split once; refuse if the output file already exists."""
    ap = argparse.ArgumentParser(description="Create the train/val/test split (once).")
    ap.add_argument("--root", default="data/training2017")
    ap.add_argument("--out", default="configs/splits.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    a = ap.parse_args(argv)

    out = Path(a.out)
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}: the split is made once and never regenerated.")
    splits = make_splits(load_reference(a.root), seed=a.seed, val=a.val, test=a.test)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out, index=False)
    print(split_table(splits).to_string())
    print(f"Saved {out} ({len(splits)} records, seed {a.seed})")


if __name__ == "__main__":
    main()
