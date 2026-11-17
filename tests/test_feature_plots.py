"""Tests for scripts/feature_plots.py on a fake features CSV and split file (never the real data)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from afdetect.data import LABELS
from afdetect.features import FEATURES

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "feature_plots.py"


def load_feature_plots():
    """Import scripts/feature_plots.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("feature_plots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_files(tmp_path):
    """Fake features (5 records per class per split) where every feature equals a split-specific value."""
    rows, split_rows = [], []
    value = {"train": 1.0, "val": 100.0, "test": 1000.0}
    for split in ("train", "val", "test"):
        for c in LABELS:
            for i in range(5):
                rec = f"{split}_{c}_{i}"
                rows.append({"record": rec, "label": c, "read_ok": 1, **{f: value[split] for f in FEATURES}})
                split_rows.append({"record": rec, "label": c, "split": split})
    rows[0]["hr_mean"] = np.nan  # a NaN must not break the plot
    pd.DataFrame(rows).to_csv(tmp_path / "features.csv", index=False)
    pd.DataFrame(split_rows).to_csv(tmp_path / "splits.csv", index=False)


def test_uses_train_only(tmp_path):
    fp = load_feature_plots()
    make_files(tmp_path)
    df = fp.load_train(tmp_path / "features.csv", tmp_path / "splits.csv")
    assert len(df) == 20 and df["record"].str.startswith("train_").all()
    table = fp.median_table(df)
    assert list(table.columns) == LABELS
    assert table.loc["n_records"].tolist() == [5, 5, 5, 5]
    assert (table.loc[FEATURES] == 1.0).all().all()  # val/test values (100, 1000) never leak in


def test_main_writes_png_and_table(tmp_path, capsys):
    fp = load_feature_plots()
    make_files(tmp_path)
    out = tmp_path / "out"
    fp.main(["--features", str(tmp_path / "features.csv"), "--splits", str(tmp_path / "splits.csv"),
             "--out", str(out)])
    assert (out / "features_by_class.png").stat().st_size > 0
    saved = pd.read_csv(out / "feature_medians_train.csv", index_col=0)
    assert list(saved.index) == ["n_records"] + FEATURES
    assert "TRAIN, n = 20" in capsys.readouterr().out
    aucs = pd.read_csv(out / "feature_auc_train.csv")
    assert set(aucs["comparison"]) == {"A_vs_N", "A_vs_O"} and len(aucs) == 2 * len(FEATURES)


def test_single_feature_auc_direction_and_ranking():
    fp = load_feature_plots()
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"label": ["A"] * n + ["N"] * n})
    for f in FEATURES:
        df[f] = rng.normal(size=2 * n)
    df.loc[df["label"] == "A", "rmssd_norm"] += 3   # strongly higher in AF
    df.loc[df["label"] == "A", "p_wave_ratio"] -= 3  # strongly lower in AF
    res = fp.single_feature_auc(df, "A", "N")
    assert set(res["feature"].head(2)) == {"rmssd_norm", "p_wave_ratio"}
    top = res.set_index("feature")
    assert top.loc["rmssd_norm", "higher_in"] == "A" and top.loc["p_wave_ratio", "higher_in"] == "N"
    assert top.loc["rmssd_norm", "separation"] > 0.9
