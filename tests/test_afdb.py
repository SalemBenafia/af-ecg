"""Tests for afdetect.afdb on a fake 250 Hz, 2-lead AFDB-style record with rhythm annotations."""
import numpy as np
import pandas as pd
import pytest
import wfdb

from afdetect.afdb import (candidate_windows, main, pick_windows, record_rng, rhythm_intervals,
                           windows_for_record)
from afdetect.features import FEATURES
from synth import make_ecg

FS = 250


def write_fake_afdb(root, name="00001", flat_at_s=None):
    """400 s record: (N 0-100 s, (AFIB 100-250 s, (AFL 250-370 s, (SVTA 370-400 s (unmapped).

    Expected 30-s windows: N 3 (0, 30, 60), A 5 (100 ... 220), O 4 (250 ... 340), SVTA none."""
    parts = [make_ecg("N", 100, FS, 1), make_ecg("A", 150, FS, 2), make_ecg("O", 120, FS, 3),
             make_ecg("N", 30, FS, 4)]
    lead0 = np.concatenate(parts)
    if flat_at_s is not None:  # make one 30-s window flat
        lead0[int(flat_at_s * FS):int((flat_at_s + 30) * FS)] = 0.0
    sig = np.column_stack([lead0, 0.5 * lead0])
    wfdb.wrsamp(name, fs=FS, units=["mV", "mV"], sig_name=["ECG1", "ECG2"], p_signal=sig, fmt=["16", "16"],
                write_dir=str(root))
    wfdb.wrann(name, "atr", sample=np.array([0, 100 * FS, 250 * FS, 370 * FS]), symbol=["+"] * 4,
               aux_note=["(N", "(AFIB", "(AFL", "(SVTA"], write_dir=str(root))
    return root / name


def test_rhythm_intervals_keep_unmapped_rhythm_separate():
    iv = rhythm_intervals([0, 100, 250, 370], ["(N", "(AFIB", "(AFL\x00", "(SVTA"], 400)
    assert iv == [(0, 100, "N"), (100, 250, "A"), (250, 370, "O"), (370, 400, None)]
    assert rhythm_intervals([0, 50], ["", "(J"], 80) == [(50, 80, "O")]  # beat notes without "(" are ignored


def test_candidate_windows_fit_inside_one_rhythm():
    wins = candidate_windows([(0, 100, "N"), (100, 250, "A"), (250, 400, None)], 30)
    assert [(a, lab) for a, _, lab in wins] == [(0, "N"), (30, "N"), (60, "N"),
                                                (100, "A"), (130, "A"), (160, "A"), (190, "A"), (220, "A")]
    assert all(b - a == 30 for a, b, _ in wins)


def test_pick_windows_caps_per_label_and_is_reproducible():
    wins = [(i * 30, i * 30 + 30, "A") for i in range(10)] + [(1000, 1030, "N")]
    picked = pick_windows(wins, 3, record_rng(0, "00001"))
    assert sum(w[2] == "A" for w in picked) == 3 and sum(w[2] == "N" for w in picked) == 1
    assert picked == pick_windows(wins, 3, record_rng(0, "00001"))
    assert picked == sorted(picked)


def test_window_counts_on_fake_record(tmp_path):
    rows, skipped = windows_for_record(write_fake_afdb(tmp_path))
    df = pd.DataFrame(rows)
    assert df["label"].value_counts().to_dict() == {"N": 3, "A": 5, "O": 4}
    assert skipped == 0
    assert df["record"].iloc[0] == "00001_0" and (df["source_record"] == "00001").all()
    assert df.loc[df["label"] == "A", "hr_mean"].notna().all()
    rows, _ = windows_for_record(tmp_path / "00001", max_per_label=2)
    assert pd.DataFrame(rows)["label"].value_counts().to_dict() == {"N": 2, "A": 2, "O": 2}


def test_flat_window_is_skipped(tmp_path):
    rows, skipped = windows_for_record(write_fake_afdb(tmp_path, flat_at_s=130))
    assert skipped == 1
    assert pd.DataFrame(rows)["label"].value_counts()["A"] == 4


def test_cli_ignores_records_without_dat(tmp_path, capsys):
    write_fake_afdb(tmp_path)
    wfdb.wrsamp("00002", fs=FS, units=["mV"], sig_name=["ECG1"], p_signal=make_ecg("N", 40, FS)[:, None],
                fmt=["16"], write_dir=str(tmp_path))
    (tmp_path / "00002.dat").unlink()
    out = tmp_path / "features" / "afdb.csv"
    main(["--root", str(tmp_path), "--out", str(out), "--jobs", "1"])
    df = pd.read_csv(out)
    assert list(df.columns) == ["record", "source_record", "label", "start_s", "read_ok"] + FEATURES
    assert len(df) == 12 and set(df["source_record"].astype(str).str.zfill(5)) == {"00001"}
    printed = capsys.readouterr().out
    assert "without .dat (not used): ['00002']" in printed


@pytest.mark.parametrize("seed", [0, 1])
def test_seed_changes_choice_only_when_capped(tmp_path, seed):
    path = write_fake_afdb(tmp_path)
    full, _ = windows_for_record(path, max_per_label=60, seed=seed)
    assert len(full) == 12  # nothing capped: every window used, whatever the seed
