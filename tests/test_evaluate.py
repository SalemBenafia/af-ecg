"""Tests for afdetect.evaluate on a tiny synthetic project in tmp_path (never the real test split or AFDB)."""
import json
import subprocess

import pandas as pd
import pytest

from afdetect import evaluate
from afdetect.train import main as train_main
from test_models import feature_table
from test_train import write_fake_project

OUTPUTS = ["summary.json", "RESULTS_table.md", "confusion_gbm.png", "predictions.csv"]


@pytest.fixture
def trained_project(tmp_path, monkeypatch):
    """Fake project with trained models; the freeze check is replaced (no git tag in tmp_path)."""
    cfg = write_fake_project(tmp_path)
    train_main(["--config", str(cfg)])
    monkeypatch.setattr(evaluate, "require_freeze", lambda: "fake-commit")
    return tmp_path, cfg


def test_evaluate_test_split_writes_every_output(trained_project):
    root, cfg = trained_project
    summary = evaluate.main(["--config", str(cfg), "--split", "test"])
    out = root / "results" / "test"
    for f in OUTPUTS:
        assert (out / f).stat().st_size > 0, f
    n_test = int((pd.read_csv(root / "splits.csv")["split"] == "test").sum())
    assert summary["run"]["n"] == n_test and summary["run"]["freeze_commit"] == "fake-commit"
    for s in summary["models"].values():
        assert s["n"] == n_test
        for m in ("challenge_F1", "AF_sensitivity", "AF_specificity"):
            lo, hi = s[f"{m}_95CI"]
            assert lo <= s[m] <= hi
    preds = pd.read_csv(out / "predictions.csv")
    assert list(preds.columns) == ["record", "label", "pred", "p_af", "confidence", "abstained"]
    ab = summary["abstain"]
    assert ab["n_kept"] == int((~preds["abstained"]).sum())
    json.loads((out / "summary.json").read_text())  # valid JSON


def test_evaluate_external_groups_by_source_record(trained_project):
    root, cfg = trained_project
    X, y = feature_table(n_per_class=30, seed=5)
    keep = y != "~"  # AFDB has only N, A, O
    ext = pd.concat([pd.DataFrame({"record": [f"w{i}" for i in range(keep.sum())],
                                   "source_record": [f"p{i % 6}" for i in range(keep.sum())],
                                   "label": y[keep], "start_s": 0.0, "read_ok": 1}),
                     X[keep].reset_index(drop=True)], axis=1)
    path = root / "fake_afdb.csv"
    ext.to_csv(path, index=False)
    summary = evaluate.main(["--config", str(cfg), "--external", str(path)])
    out = root / "results" / "fake_afdb"
    for f in OUTPUTS:
        assert (out / f).exists(), f
    assert summary["run"]["kind"] == "external" and summary["run"]["n_groups"] == 6
    assert "source_record" in pd.read_csv(out / "predictions.csv").columns
    assert "Main numbers: AF sensitivity and specificity" in (out / "RESULTS_table.md").read_text()


def test_external_csv_must_have_all_columns(tmp_path):
    pd.DataFrame({"record": ["a"], "label": ["N"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        evaluate.load_external(tmp_path / "bad.csv")


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_require_freeze_in_a_git_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    git(tmp_path, "init", "-q")
    git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "gbm.joblib").write_text("v1")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "models")
    with pytest.raises(SystemExit, match="does not exist"):
        evaluate.require_freeze()
    git(tmp_path, "tag", "v1-frozen")
    assert len(evaluate.require_freeze()) == 40  # tag commit hash
    (tmp_path / "models" / "gbm.joblib").write_text("changed after the freeze")
    with pytest.raises(SystemExit, match="changed since"):
        evaluate.require_freeze()


def test_cli_refuses_without_freeze(tmp_path, monkeypatch):
    """Without the tag, evaluate stops before reading any data."""
    cfg = write_fake_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    git(tmp_path, "init", "-q")
    with pytest.raises(SystemExit, match="Refusing to evaluate"):
        evaluate.main(["--config", str(cfg), "--split", "test"])
    assert not (tmp_path / "results" / "test").exists()
