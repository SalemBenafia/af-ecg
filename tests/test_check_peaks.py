"""Tests for scripts/check_peaks.py on a tiny synthetic 4-class dataset."""
import importlib.util
from pathlib import Path

import numpy as np
import wfdb

from afdetect.data import LABELS, load_reference
from synth import make_ecg

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_peaks.py"


def load_check_peaks():
    """Import scripts/check_peaks.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("check_peaks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_synth_dataset(root, per_class=2, seconds=15, fs=300):
    """Write synthetic WFDB records (per_class per label) and a REFERENCE.csv."""
    lines = []
    for i, label in enumerate(LABELS):
        for j in range(per_class):
            name = f"S{i}{j:03d}"
            x = make_ecg(label, seconds, fs, seed=10 * i + j)
            wfdb.wrsamp(name, fs=fs, units=["mV"], sig_name=["ECG"], p_signal=x[:, None], fmt=["16"],
                        write_dir=str(root))
            lines.append(f"{name},{label}\n")
    (root / "REFERENCE.csv").write_text("".join(lines))
    # all records in train, plus one extra test record that must never be picked
    rows = [ln.strip().split(",") for ln in lines]
    (root / "splits.csv").write_text("record,label,split\n" + "".join(f"{r},{lab},train\n" for r, lab in rows)
                                     + "X99999,N,test\n")


def test_peak_summary():
    cp = load_check_peaks()
    s = cp.peak_summary(np.arange(0, 3000, 300), 3000)  # 1 beat per second for 10 s
    assert s["n_peaks"] == 10 and s["hr"] == 60.0 and s["frac_rr_implausible"] == 0.0
    empty = cp.peak_summary(np.array([], dtype=int), 3000)
    assert empty["n_peaks"] == 0 and np.isnan(empty["hr"])


def test_main_saves_png_and_summaries(tmp_path, capsys):
    cp = load_check_peaks()
    data = tmp_path / "data"
    data.mkdir()
    make_synth_dataset(data)
    summaries = cp.main(["--root", str(data), "--out", str(tmp_path / "out"), "--seed", "0",
                         "--splits", str(data / "splits.csv")])
    assert len(summaries) == 8
    assert {s["label"] for s in summaries.values()} == set(LABELS)
    normal = [s for s in summaries.values() if s["label"] == "N"]
    assert all(50 <= s["hr"] <= 100 for s in normal)
    assert (tmp_path / "out" / "peaks.png").stat().st_size > 0
    assert "Saved" in capsys.readouterr().out


def test_restrict_to_split_drops_other_splits(tmp_path):
    cp = load_check_peaks()
    make_synth_dataset(tmp_path, per_class=1, seconds=10)
    splits = tmp_path / "splits.csv"
    splits.write_text("record,label,split\nS0000,N,train\nS1000,A,test\nS2000,O,val\nS3000,~,train\n")
    ref = load_reference(tmp_path)
    assert set(cp.restrict_to_split(ref, splits, "train")["record"]) == {"S0000", "S3000"}
    assert len(cp.restrict_to_split(ref, splits, "all")) == 4
