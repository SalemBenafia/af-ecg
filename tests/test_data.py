"""Tests for afdetect.data, using fake WFDB records and CSV files written to tmp_path."""
import shutil

import numpy as np
import pytest
import wfdb

from afdetect.data import LABELS, NAMES, load_reference, read_csv_ecg, read_uploaded, read_wfdb

FS = 300


def fake_signal(n=FS * 10):
    """Two-lead test signal in mV: a 1.2 Hz sine and a scaled, shifted copy."""
    t = np.arange(n) / FS
    lead0 = 0.8 * np.sin(2 * np.pi * 1.2 * t)
    return np.column_stack([lead0, 0.5 * lead0 + 0.1])


def write_record(folder, name="rec1", signal=None, units=("mV", "mV")):
    """Write a 2-lead WFDB record (.hea + .dat) and return the path without extension."""
    signal = fake_signal() if signal is None else signal
    wfdb.wrsamp(name, fs=FS, units=list(units), sig_name=["ECG1", "ECG2"], p_signal=signal,
                fmt=["16", "16"], write_dir=str(folder))
    return folder / name


def test_labels_and_names():
    assert LABELS == ["N", "A", "O", "~"]
    assert set(NAMES) == set(LABELS)


def test_load_reference(tmp_path):
    (tmp_path / "REFERENCE.csv").write_text("A00001,N\nA00002,A\nA00003,O\nA00004,~\n")
    ref = load_reference(tmp_path)
    assert list(ref.columns) == ["record", "label"]
    assert ref["record"].tolist() == ["A00001", "A00002", "A00003", "A00004"]
    assert ref["label"].tolist() == LABELS


def test_load_reference_rejects_unknown_label(tmp_path):
    (tmp_path / "REFERENCE.csv").write_text("A00001,N\nA00002,X\n")
    with pytest.raises(AssertionError, match="X"):
        load_reference(tmp_path)


def test_read_wfdb_values_fs_and_channel(tmp_path):
    sig = fake_signal()
    path = write_record(tmp_path, signal=sig)
    x0, fs = read_wfdb(path)
    x1, _ = read_wfdb(path, channel=1)
    assert fs == FS
    assert x0.dtype == np.float64 and len(x0) == len(sig)
    np.testing.assert_allclose(x0, sig[:, 0], atol=1e-3)
    np.testing.assert_allclose(x1, sig[:, 1], atol=1e-3)


def test_read_wfdb_converts_microvolts(tmp_path):
    sig_uv = fake_signal() * 1000  # same signal, stored in µV
    path = write_record(tmp_path, signal=sig_uv, units=("uV", "uV"))
    x, _ = read_wfdb(path)
    np.testing.assert_allclose(x, sig_uv[:, 0] / 1000, atol=1e-3)


def test_read_csv_ecg_without_and_with_header(tmp_path):
    values = [0.1, -0.2, 0.3, 0.05]
    plain = tmp_path / "plain.csv"
    plain.write_text("\n".join(map(str, values)) + "\n")
    headed = tmp_path / "headed.csv"
    headed.write_text("ecg_mV\n" + "\n".join(map(str, values)) + "\n")
    for path in (plain, headed):
        x, fs = read_csv_ecg(path, 250)
        np.testing.assert_allclose(x, values)
        assert fs == 250.0


def test_read_csv_ecg_rejects_bad_input(tmp_path):
    two_cols = tmp_path / "two.csv"
    two_cols.write_text("0.1,0.2\n0.3,0.4\n")
    with pytest.raises(ValueError, match="one column"):
        read_csv_ecg(two_cols, 300)
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_csv_ecg(empty, 300)
    with pytest.raises(ValueError, match="sampling rate"):
        read_csv_ecg(two_cols, 0)


def test_read_uploaded_csv(tmp_path):
    path = tmp_path / "ecg.csv"
    path.write_text("0.1\n0.2\n0.3\n")
    x, fs = read_uploaded([str(path)], 500)
    assert fs == 500.0 and len(x) == 3


def test_read_uploaded_wfdb_pair_in_separate_folders(tmp_path):
    """Gradio stores each upload in its own folder: the pair must still be read together."""
    src_dir, up_hea, up_dat = tmp_path / "src", tmp_path / "upload1", tmp_path / "upload2"
    for folder in (src_dir, up_hea, up_dat):
        folder.mkdir()
    src = write_record(src_dir)
    shutil.copy(src.with_suffix(".hea"), up_hea)
    shutil.copy(src.with_suffix(".dat"), up_dat)
    x, fs = read_uploaded([str(up_hea / "rec1.hea"), str(up_dat / "rec1.dat")], 300)
    assert fs == FS
    np.testing.assert_allclose(x, fake_signal()[:, 0], atol=1e-3)


def test_read_uploaded_rejects_incomplete_or_unknown_files(tmp_path):
    path = write_record(tmp_path)
    with pytest.raises(ValueError, match="Missing the signal file"):
        read_uploaded([str(path.with_suffix(".hea"))], 300)
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    with pytest.raises(ValueError, match="Upload a .csv"):
        read_uploaded([str(txt)], 300)
