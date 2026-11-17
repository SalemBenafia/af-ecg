"""Tests for app.py with a tiny gbm trained on synthetic ECG features in tmp_path (never the real models)."""
import importlib.util
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import wfdb

from afdetect.features import extract_features
from afdetect.models import make_model
from synth import KINDS, make_ecg

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def tiny_gbm():
    """A gbm fit on features of 12 synthetic ECGs per class (fit once per test module)."""
    rows = [{"label": k, **extract_features(make_ecg(k, 30, 300, seed=s), 300)} for k in KINDS for s in range(12)]
    df = pd.DataFrame(rows)
    model, cols = make_model("gbm", seed=0)
    model.fit(df[cols], df["label"])
    return {"model": model, "cols": cols}


@pytest.fixture
def app(tmp_path, monkeypatch, tiny_gbm):
    """app.py loaded with models and an AF example from tmp_path."""
    (tmp_path / "models").mkdir()
    joblib.dump(tiny_gbm, tmp_path / "models" / "gbm.joblib")
    (tmp_path / "models" / "abstain.json").write_text(json.dumps({"threshold": 0.30}))
    (tmp_path / "examples").mkdir()
    pd.DataFrame({"ecg_mV": make_ecg("A", 30, 300, seed=99)}).to_csv(tmp_path / "examples" / "AF_1.csv", index=False)
    monkeypatch.setenv("AFDETECT_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("app_under_test", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_af_example_is_recognised(app):
    probs, text, fig = app.analyze(None, 300, "AF_1")
    assert set(probs) == {"Normal rhythm", "Atrial fibrillation", "Other rhythm", "Too noisy to classify"}
    assert max(probs, key=probs.get) == "Atrial fibrillation"
    assert "## Atrial fibrillation" in text and "clinician should review" in text
    assert len(fig.axes) == 3


def test_uploaded_wfdb_pair_of_normal_ecg(app, tmp_path):
    """Gradio stores every upload in its own folder: the .hea and .dat arrive in different folders."""
    src = tmp_path / "src_rec"
    src.mkdir()
    wfdb.wrsamp("rec", fs=300, units=["mV"], sig_name=["ECG"], p_signal=make_ecg("N", 30, 300, seed=42)[:, None],
                fmt=["16"], write_dir=str(src))
    paths = []
    for i, ext in enumerate((".hea", ".dat")):
        folder = tmp_path / f"upload{i}"
        folder.mkdir()
        shutil.copy(src / f"rec{ext}", folder)
        paths.append(str(folder / f"rec{ext}"))
    probs, text, _ = app.analyze(paths, 300, "(none)")
    assert "## Normal rhythm" in text
    assert "Average heart rate: " in text and "not measurable" not in text.split("Average heart rate:")[1][:20]


def test_short_csv_is_refused(app, tmp_path):
    path = tmp_path / "short.csv"
    pd.DataFrame({"ecg_mV": make_ecg("N", 5, 300)}).to_csv(path, index=False)
    with pytest.raises(Exception, match="too short"):
        app.analyze([str(path)], 300, "(none)")


def test_no_input_is_refused(app):
    with pytest.raises(Exception, match="Upload a recording"):
        app.analyze(None, 300, "(none)")


def test_uncertain_below_threshold_and_disclaimer(app):
    f = {"hr_mean": np.nan, "rmssd_norm": 0.1, "p_wave_ratio": np.nan, "tmpl_corr": 0.9}
    text = app.verdict("N", np.array([0.28, 0.26, 0.24, 0.22]), f)
    assert text.startswith("## Uncertain") and "not measurable" in text
    assert "Not a medical device" in app.DISCLAIMER and "Uncertain" in text
