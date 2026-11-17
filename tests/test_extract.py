"""Tests for afdetect.extract on fake WFDB records in tmp_path."""
import numpy as np
import pandas as pd
import wfdb

from afdetect.extract import extract_all, features_for_record, main, report
from afdetect.features import FEATURES
from synth import make_ecg


def make_dataset(root):
    """Three synthetic records (N, A, O) plus one listed record whose signal file is missing."""
    lines = []
    for i, label in enumerate("NAO"):
        name = f"A0000{i + 1}"
        x = make_ecg(label, 20, 300, seed=i)
        wfdb.wrsamp(name, fs=300, units=["mV"], sig_name=["ECG"], p_signal=x[:, None], fmt=["16"],
                    write_dir=str(root))
        lines.append(f"{name},{label}\n")
    # A broken record: header only, the .dat file is gone.
    wfdb.wrsamp("A00004", fs=300, units=["mV"], sig_name=["ECG"], p_signal=make_ecg("N", 10)[:, None],
                fmt=["16"], write_dir=str(root))
    (root / "A00004.dat").unlink()
    lines.append("A00004,~\n")
    (root / "REFERENCE.csv").write_text("".join(lines))


def test_features_for_record_ok_and_failure(tmp_path):
    make_dataset(tmp_path)
    good = features_for_record(tmp_path, "A00001")
    assert good["read_ok"] == 1 and good["n_beats"] > 10
    bad = features_for_record(tmp_path, "A00004")
    assert bad["read_ok"] == 0 and all(np.isnan(bad[k]) for k in FEATURES)


def test_cli_keeps_every_record_and_merges_labels(tmp_path, capsys):
    make_dataset(tmp_path)
    out = tmp_path / "features" / "cinc.csv"
    main(["--root", str(tmp_path), "--out", str(out), "--jobs", "2"])
    df = pd.read_csv(out)
    assert list(df.columns) == ["record", "label", "read_ok"] + FEATURES
    assert df["record"].tolist() == ["A00001", "A00002", "A00003", "A00004"]  # same order, none missing
    assert df["label"].tolist() == ["N", "A", "O", "~"]
    assert df.loc[3, FEATURES].isna().all() and df.loc[3, "read_ok"] == 0
    assert df.loc[:2, "n_beats"].notna().all()
    printed = capsys.readouterr().out
    assert "Unreadable records: 1" in printed and "NaN values per feature" in printed


def test_limit_and_report(tmp_path):
    make_dataset(tmp_path)
    out = tmp_path / "f.csv"
    df = main(["--root", str(tmp_path), "--out", str(out), "--jobs", "1", "--limit", "2"])
    assert len(df) == 2 and len(pd.read_csv(out)) == 2
    text = report(df)
    assert "Records: 2" in text and "N 1, A 1, O 0, ~ 0" in text


def test_extract_all_preserves_reference_order(tmp_path):
    make_dataset(tmp_path)
    ref = pd.DataFrame({"record": ["A00003", "A00001"], "label": ["O", "N"]})
    df = extract_all(tmp_path, ref, jobs=1)
    assert df["record"].tolist() == ["A00003", "A00001"] and df["label"].tolist() == ["O", "N"]
