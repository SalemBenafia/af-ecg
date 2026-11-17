"""Data IO: CinC 2017 reference labels, WFDB records, CSV and uploaded ECG files."""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

LABELS = ["N", "A", "O", "~"]
NAMES = {
    "N": "Normal rhythm",
    "A": "Atrial fibrillation",
    "O": "Other rhythm",
    "~": "Too noisy to classify",
}

# Factors that convert a WFDB physical unit to millivolts.
_TO_MV = {"mv": 1.0, "uv": 1e-3, "µv": 1e-3, "v": 1e3}


def load_reference(root):
    """Read root/REFERENCE.csv (no header) into a DataFrame with columns record, label.

    Fails loudly if any label is not one of LABELS (N, A, O, ~)."""
    ref = pd.read_csv(Path(root) / "REFERENCE.csv", header=None, names=["record", "label"],
                      dtype=str, keep_default_na=False)
    ref["record"] = ref["record"].str.strip()
    ref["label"] = ref["label"].str.strip()
    unexpected = sorted(set(ref["label"]) - set(LABELS))
    assert not unexpected, f"Unexpected labels in REFERENCE.csv: {unexpected}"
    return ref


def read_wfdb(path, channel=0):
    """Read one channel of a WFDB record (path without extension).

    Returns (signal in mV as float64, sampling rate in Hz)."""
    rec = wfdb.rdrecord(str(path), channels=[channel])
    unit = (rec.units[0] or "mV").strip().lower()
    if unit not in _TO_MV:
        raise ValueError(f"Unsupported signal unit {rec.units[0]!r}: expected mV, uV or V")
    signal = rec.p_signal[:, 0].astype(np.float64) * _TO_MV[unit]
    return signal, float(rec.fs)


def read_csv_ecg(path, fs):
    """Read a one-column CSV ECG (optional header row) sampled at fs Hz.

    Returns (signal as float64, fs)."""
    if fs is None or float(fs) <= 0:
        raise ValueError("The sampling rate must be a positive number of Hz.")
    try:
        df = pd.read_csv(path, header=None)
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.") from None
    if not pd.api.types.is_numeric_dtype(df[0]):
        df = pd.read_csv(path, header=0)  # first row was a header
    if df.shape[1] != 1:
        raise ValueError(f"Expected one column of ECG samples, found {df.shape[1]} columns.")
    if len(df) == 0 or not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        raise ValueError("The CSV column must contain numeric ECG samples.")
    return df.iloc[:, 0].to_numpy(dtype=np.float64), float(fs)


def read_uploaded(paths, csv_fs):
    """Read an uploaded recording for the app: one .csv, or a .hea plus its .mat/.dat file.

    Upload tools (Gradio) put each file in its own folder, so the WFDB pair is copied into
    one temporary folder under its original names before reading."""
    paths = [Path(p) for p in paths]
    by_ext = {}
    for p in paths:
        by_ext.setdefault(p.suffix.lower(), []).append(p)
    if ".csv" in by_ext:
        return read_csv_ecg(by_ext[".csv"][0], csv_fs)
    if ".hea" not in by_ext:
        raise ValueError("Upload a .csv file, or a .hea file together with its .mat or .dat file.")
    hea = by_ext[".hea"][0]
    data_files = [p for p in by_ext.get(".mat", []) + by_ext.get(".dat", []) if p.stem == hea.stem]
    if not data_files:
        raise ValueError(f"Missing the signal file for {hea.name}: upload {hea.stem}.mat or {hea.stem}.dat too.")
    with tempfile.TemporaryDirectory() as tmp:
        for p in [hea, *data_files]:
            shutil.copy(p, Path(tmp) / p.name)
        return read_wfdb(Path(tmp) / hea.stem)
