"""Tests for scripts/explore.py on a tiny fake 4-class dataset."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from afdetect.data import LABELS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "explore.py"
FS = 300


def load_explore():
    """Import scripts/explore.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("explore", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fake_dataset(root, seconds=(12, 30, 45)):
    """Write 3 single-lead records per class plus REFERENCE.csv; return the reference rows."""
    rows, rng = [], np.random.default_rng(0)
    for i, label in enumerate(LABELS):
        for j, sec in enumerate(seconds):
            name = f"A{i}{j:04d}"
            sig = 0.5 * np.sin(2 * np.pi * 1.1 * np.arange(sec * FS) / FS) + 0.01 * rng.standard_normal(sec * FS)
            wfdb.wrsamp(name, fs=FS, units=["mV"], sig_name=["ECG"], p_signal=sig[:, None], fmt=["16"],
                        write_dir=str(root))
            rows.append((name, label))
    (root / "REFERENCE.csv").write_text("".join(f"{r},{lab}\n" for r, lab in rows))
    return pd.DataFrame(rows, columns=["record", "label"])


def test_helpers(tmp_path):
    explore = load_explore()
    ref = make_fake_dataset(tmp_path)
    counts = explore.class_counts(ref)
    assert counts["count"].tolist() == [3, 3, 3, 3]
    lens = explore.recording_lengths(tmp_path, ref)
    assert sorted(set(lens["seconds"])) == [12.0, 30.0, 45.0]
    assert explore.length_bins(lens["seconds"]) == {"< 30 s": 4, "= 30 s": 4, "> 30 s": 4}
    picks = explore.pick_examples(ref, seed=0)
    assert all(len(set(picks[c])) == 2 for c in LABELS)
    assert picks == explore.pick_examples(ref, seed=0)  # same seed, same picks


def test_main_prints_counts_and_saves_png(tmp_path, capsys):
    explore = load_explore()
    data = tmp_path / "data"
    data.mkdir()
    make_fake_dataset(data)
    out = tmp_path / "out"
    explore.main(["--root", str(data), "--out", str(out), "--seed", "1"])
    printed = capsys.readouterr().out
    assert "Recordings: 12" in printed
    assert "Atrial fibrillation" in printed
    assert (out / "examples.png").stat().st_size > 0
