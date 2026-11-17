"""Tests for scripts/make_examples.py on a fake project (never the real test split)."""
import importlib.util
from pathlib import Path

import pandas as pd
import wfdb

from afdetect.data import read_csv_ecg
from synth import make_ecg

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_examples.py"


def load_script():
    """Import scripts/make_examples.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("make_examples", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_project(root):
    """4 records per class per split (train/test) as WFDB files at 250 Hz, a split file and test predictions."""
    rows, preds = [], []
    for split in ("train", "test"):
        for label in ("N", "A", "O", "~"):
            for i in range(4):
                rec = f"{split[0].upper()}{'NAO~'.index(label)}{i:03d}"
                wfdb.wrsamp(rec, fs=250, units=["mV"], sig_name=["ECG"],
                            p_signal=make_ecg(label, 12, 250, seed=i)[:, None], fmt=["16"], write_dir=str(root))
                rows.append({"record": rec, "label": label, "split": split})
                if split == "test":
                    preds.append({"record": rec, "label": label, "pred": "N"})
    pd.DataFrame(rows).to_csv(root / "splits.csv", index=False)
    pd.DataFrame(preds).to_csv(root / "predictions.csv", index=False)


def test_six_test_examples_reproducible(tmp_path):
    me = load_script()
    fake_project(tmp_path)
    args = ["--root", str(tmp_path), "--splits", str(tmp_path / "splits.csv"),
            "--predictions", str(tmp_path / "predictions.csv"), "--out", str(tmp_path / "examples")]
    rows = me.main(args)
    names = sorted(p.name for p in (tmp_path / "examples").glob("*.csv"))
    assert names == ["AF_1.csv", "AF_2.csv", "Noisy_1.csv", "Normal_1.csv", "Normal_2.csv", "Other_1.csv"]
    assert all(r["record"].startswith("T") for r in rows)  # TEST records only
    x, fs = read_csv_ecg(tmp_path / "examples" / "AF_1.csv", 300)
    assert fs == 300 and len(x) == 12 * 300  # resampled from 250 to 300 Hz
    readme = (tmp_path / "examples" / "README.md").read_text()
    assert all(r["record"] in readme for r in rows) and "(wrong)" in readme  # the fake model says N for all
    assert [r["record"] for r in me.main(args)] == [r["record"] for r in rows]  # same seed, same picks
