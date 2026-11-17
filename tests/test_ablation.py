"""Tests for scripts/ablation.py on the fake project from test_train (never the real data)."""
import importlib.util
from pathlib import Path

import pandas as pd

from afdetect.metrics import summarize
from afdetect.train import load_config, load_split
from test_train import write_fake_project

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ablation.py"


def load_ablation():
    """Import scripts/ablation.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("ablation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_sets_are_nested():
    ab = load_ablation()
    sets = list(ab.FEATURE_SETS.values())
    assert [len(s) for s in sets] == [11, 12, 13, 21]
    assert all(set(a) < set(b) for a, b in zip(sets, sets[1:]))


def test_main_writes_outputs_and_keeps_recommendation(tmp_path):
    ab = load_ablation()
    cfg = write_fake_project(tmp_path)
    out = tmp_path / "val"
    agg, imp = ab.main(["--config", str(cfg), "--out", str(out), "--seeds", "0", "1", "--n-repeats", "2"])
    runs = pd.read_csv(out / "ablation.csv")
    assert len(runs) == 4 * 2 * 2 * 2  # feature sets x class weights x learning rates x seeds
    assert len(agg) == 16 and (agg["n_seeds"] == 2).all()
    assert agg["F1_mean"].is_monotonic_decreasing
    assert len(imp) == agg.iloc[0]["n_features"]
    md = (out / "ablation.md").read_text()
    assert "| Feature set |" in md and "Permutation importance" in md and "## Recommendation" in md

    # A hand-written recommendation survives a re-run.
    (out / "ablation.md").write_text(md.replace("(to be written after reading the table)", "Keep gbm."))
    ab.main(["--config", str(cfg), "--out", str(out), "--seeds", "0", "--n-repeats", "2"])
    assert "Keep gbm." in (out / "ablation.md").read_text()


def test_scorer_matches_summarize(tmp_path):
    ab = load_ablation()
    cfg = load_config(write_fake_project(tmp_path))
    tr, va = load_split(cfg, "train"), load_split(cfg, "val")
    cols = ab.FEATURE_SETS["RR only"]
    model = ab.fit_gbm(tr, cols, "balanced", 0.1, seed=0)
    assert ab.challenge_f1_scorer(model, va[cols], va["label"]) == summarize(
        va["label"], model.predict(va[cols]))["challenge_F1"]
