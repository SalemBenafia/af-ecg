"""Tests for scripts/errors.py on fake predictions, features and synthetic WFDB records."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from synth import make_ecg

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "errors.py"


def load_errors():
    """Import scripts/errors.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("errors_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fake(tmp_path, n=60):
    """Fake predictions + features for n records, and synthetic WFDB files for every record."""
    rng = np.random.default_rng(0)
    labels = np.array(["N", "A", "O", "~"] * (n // 4))
    preds = labels.copy()
    preds[1::8] = "N"   # some AF missed
    preds[2::8] = "A"   # some Other called AF
    preds[5::8] = "O"   # some AF called Other
    records = [f"T{i:04d}" for i in range(n)]
    pd.DataFrame({"record": records, "label": labels, "pred": preds, "p_af": rng.random(n),
                  "confidence": rng.uniform(0.3, 1, n), "abstained": False}).to_csv(tmp_path / "pred.csv", index=False)
    feats = pd.DataFrame({"record": records, "duration_s": rng.choice([15.0, 30.0, 55.0], n),
                          "hr_mean": rng.choice([50.0, 75.0, 120.0], n), "nk_quality": rng.random(n)})
    feats.loc[0, "hr_mean"] = np.nan
    feats.to_csv(tmp_path / "feats.csv", index=False)
    for r, lab in zip(records, labels):
        wfdb.wrsamp(r, fs=300, units=["mV"], sig_name=["ECG"], p_signal=make_ecg(lab, 20, 300, 1)[:, None],
                    fmt=["16"], write_dir=str(tmp_path))
    return tmp_path


def test_groups_and_table(tmp_path):
    er = load_errors()
    make_fake(tmp_path)
    df = er.add_groups(er.load(tmp_path / "pred.csv", tmp_path / "feats.csv"))
    assert set(df["length"]) == {"< 20 s", "20-40 s", ">= 40 s"}
    assert "unknown (NaN)" in set(df["heart_rate"])
    table = er.subgroup_table(df, n_boot=50)
    for by in ("length", "heart_rate", "quality"):
        assert table.loc[table["by"] == by, "n"].sum() == 60  # every record in exactly one group
    assert table["descriptive_only"].all()  # all groups are tiny here
    assert "descriptive only" in er.table_markdown(table)


def test_pick_errors_kinds():
    er = load_errors()
    df = pd.DataFrame({"record": list("abcdefg"), "label": ["A", "A", "N", "O", "A", "A", "N"],
                       "pred": ["N", "~", "A", "A", "O", "O", "N"],
                       "p_af": [0.2, 0.1, 0.9, 0.95, 0.3, 0.05, 0.1]})
    picks = er.pick_errors(df)
    assert [(k, r["record"]) for k, r in picks] == [("af_missed", "b"), ("af_missed", "a"),
                                                    ("false_af_alarm", "d"), ("false_af_alarm", "c"),
                                                    ("af_called_other", "f"), ("af_called_other", "e")]


def test_main_writes_tables_and_six_plots(tmp_path):
    er = load_errors()
    make_fake(tmp_path)
    out = tmp_path / "errors"
    er.main(["--predictions", str(tmp_path / "pred.csv"), "--features", str(tmp_path / "feats.csv"),
             "--root", str(tmp_path), "--out", str(out), "--n-boot", "50"])
    assert (out / "subgroups.csv").exists() and (out / "subgroups.md").exists()
    assert len(list(out.glob("*.png"))) == 6
