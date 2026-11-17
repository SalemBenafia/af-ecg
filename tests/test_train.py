"""End-to-end test of afdetect.train on fake features and a fake split (never the real data)."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from afdetect.data import LABELS
from afdetect.train import choose_abstain_threshold, load_config, load_split, main
from test_models import feature_table


def write_fake_project(tmp_path):
    """Fake features CSV + split file (60% train, 20% val, 20% test) + config; returns the config path."""
    X, y = feature_table(n_per_class=60, seed=1)
    records = [f"R{i:04d}" for i in range(len(y))]
    feats = pd.concat([pd.DataFrame({"record": records, "label": y, "read_ok": 1}), X], axis=1)
    feats.to_csv(tmp_path / "features.csv", index=False)
    split = np.array(["train", "train", "train", "val", "test"] * (len(y) // 5))
    pd.DataFrame({"record": records, "label": y, "split": split}).to_csv(tmp_path / "splits.csv", index=False)
    cfg = {"seed": 0, "features_csv": str(tmp_path / "features.csv"), "split_file": str(tmp_path / "splits.csv"),
           "models": ["rr_rule", "logreg_rr", "gbm"], "models_dir": str(tmp_path / "models"),
           "out_dir": str(tmp_path / "results"), "n_boot": 50, "abstain_min_coverage_pct": 90,
           "abstain_grid": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]}
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_main_end_to_end(tmp_path, capsys):
    cfg_path = write_fake_project(tmp_path)
    summary = main(["--config", str(cfg_path)])
    for name in ("rr_rule", "logreg_rr", "gbm"):
        assert (tmp_path / "models" / f"{name}.joblib").exists()
    assert summary["run"]["n_val"] == 48 and summary["run"]["n_train"] == 144
    for s in summary["models"].values():
        assert s["n"] == 48  # only VAL records were scored
        lo, hi = s["challenge_F1_95CI"]
        assert lo <= s["challenge_F1"] <= hi
        assert np.array(s["confusion"]).shape == (4, 4)
    abstain = json.loads((tmp_path / "models" / "abstain.json").read_text())
    assert abstain["chosen_on"] == "val"
    at = [r for r in abstain["curve"] if r["threshold"] == abstain["threshold"]]
    assert abstain["fallback"] or at[0]["kept_pct"] >= 90
    saved = json.loads((tmp_path / "results" / "val" / "summary.json").read_text())  # valid JSON, no NaN
    assert set(saved["models"]) == {"rr_rule", "logreg_rr", "gbm"}
    assert "| gbm |" in (tmp_path / "results" / "val" / "VAL_table.md").read_text()
    assert "abstain threshold" in capsys.readouterr().out


def test_load_split_refuses_test(tmp_path):
    cfg = load_config(write_fake_project(tmp_path))
    with pytest.raises(ValueError, match="may only load"):
        load_split(cfg, "test")
    assert set(load_split(cfg, "val")["label"]) == set(LABELS)


def test_load_split_rejects_label_mismatch(tmp_path):
    cfg = load_config(write_fake_project(tmp_path))
    splits = pd.read_csv(cfg["split_file"])
    splits.loc[0, "label"] = "A" if splits.loc[0, "label"] != "A" else "N"
    splits.to_csv(cfg["split_file"], index=False)
    with pytest.raises(ValueError, match="Labels differ"):
        load_split(cfg, "train")


def test_choose_abstain_threshold():
    curve = [{"threshold": 0.3, "kept_pct": 100.0}, {"threshold": 0.5, "kept_pct": 92.0},
             {"threshold": 0.6, "kept_pct": 85.0}]
    assert choose_abstain_threshold(curve, 90) == {"threshold": 0.5, "kept_pct_at_threshold": 92.0,
                                                   "fallback": False}
    fallback = choose_abstain_threshold(curve, 101)
    assert fallback["threshold"] == 0.0 and fallback["fallback"] is True
