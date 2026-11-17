"""Pick the 6 demo examples from the TEST split (seed 7) and save them as one-column 300 Hz CSV files.

Usage: python scripts/make_examples.py --root data/training2017 --splits configs/splits.csv --out examples
Picks 2 Normal, 2 AF, 1 Other and 1 Noisy recording. The picks are fixed by the seed and are never re-rolled
to get nicer examples: if the model gets one wrong, the README says so.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from afdetect.data import NAMES, read_wfdb
from afdetect.features import FS, to_fs

SEED = 7
PLAN = [("N", "Normal", 2), ("A", "AF", 2), ("O", "Other", 1), ("~", "Noisy", 1)]


def pick_examples(splits, seed=SEED):
    """[(file stem, record, label)] drawn from the sorted TEST records of each label with a fixed seed."""
    test = splits[splits["split"] == "test"]
    rng = np.random.default_rng(seed)
    picks = []
    for label, stem, k in PLAN:
        records = np.sort(test.loc[test["label"] == label, "record"].to_numpy())
        for i, rec in enumerate(rng.choice(records, size=k, replace=False), 1):
            picks.append((f"{stem}_{i}", str(rec), label))
    return picks


def model_answers(predictions_csv):
    """record -> the frozen gbm's predicted label on the test split, if the predictions file exists."""
    path = Path(predictions_csv)
    if not path.exists():
        return {}
    preds = pd.read_csv(path)
    return dict(zip(preds["record"], preds["pred"]))


def readme(rows):
    """examples/README.md: which file is which record, its true label and the model's answer."""
    lines = ["# Demo examples", "",
             "Six recordings from the **test split** of the PhysioNet/CinC Challenge 2017 training set (AliveCor "
             "single-lead ECG, open access, ODC Attribution licence [ADD SOURCE]), picked with a fixed seed "
             f"({SEED}) and saved as one column of samples in mV at {FS} Hz. They were not hand-picked and are "
             "never re-drawn to get nicer examples.", "",
             "| File | Record | True label | Length (s) | Frozen gbm answer (results/test/predictions.csv) |",
             "|---|---|---|---|---|"]
    for r in rows:
        answer = NAMES.get(r["model_pred"], "not available")
        mark = "" if r["model_pred"] in (None, r["label"]) else " (wrong)"
        lines.append(f"| {r['file']} | {r['record']} | {NAMES[r['label']]} | {r['seconds']:.1f} | {answer}{mark} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Write the example CSVs and their README."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/training2017")
    ap.add_argument("--splits", default="configs/splits.csv")
    ap.add_argument("--predictions", default="results/test/predictions.csv")
    ap.add_argument("--out", default="examples")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    answers = model_answers(a.predictions)
    rows = []
    for stem, record, label in pick_examples(pd.read_csv(a.splits, dtype=str), a.seed):
        x, fs = read_wfdb(Path(a.root) / record)
        ecg = to_fs(x, fs)
        pd.DataFrame({"ecg_mV": np.round(ecg, 4)}).to_csv(out / f"{stem}.csv", index=False)
        rows.append({"file": f"{stem}.csv", "record": record, "label": label, "seconds": len(ecg) / FS,
                     "model_pred": answers.get(record)})
    (out / "README.md").write_text(readme(rows))
    for r in rows:
        print(f"{r['file']:<13} {r['record']}  true {r['label']}  model {r['model_pred']}  {r['seconds']:.1f} s")
    print(f"Saved {len(rows)} examples and README.md in {out}/")
    return rows


if __name__ == "__main__":
    main()
