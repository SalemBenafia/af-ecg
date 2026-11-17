"""CLI: fit models on TRAIN, compare on VAL, choose the abstain threshold.

Usage: python -m afdetect.train --config configs/train.yaml
Writes models/<name>.joblib, models/abstain.json, results/val/summary.json and results/val/VAL_table.md.
The test split is never loaded here (load_split refuses it).
"""
import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from .metrics import bootstrap_ci, confusion, coverage_curve, summarize
from .models import make_model, predict

ALLOWED_SPLITS = ("train", "val")


def load_config(path):
    """Read the YAML training config."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_split(cfg, split):
    """Feature rows of one split (train or val only), after checking features and split file agree."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"train.py may only load {ALLOWED_SPLITS}, not {split!r}")
    df = labelled_features(cfg)
    return df[df["split"] == split].reset_index(drop=True)


def labelled_features(cfg):
    """All feature rows with their split column, after checking that features and split file agree."""
    feats = pd.read_csv(cfg["features_csv"])
    splits = pd.read_csv(cfg["split_file"], dtype=str)
    if splits["record"].duplicated().any():
        raise ValueError("A record appears more than once in the split file.")
    df = feats.merge(splits[["record", "label", "split"]], on="record", suffixes=("", "_split"),
                     validate="one_to_one")
    mismatch = df["label"] != df["label_split"]
    if mismatch.any():
        raise ValueError(f"Labels differ between features and split file for {int(mismatch.sum())} records.")
    return df.drop(columns="label_split")


def choose_abstain_threshold(curve, min_coverage_pct):
    """Highest threshold whose kept_pct >= min_coverage_pct; 0.0 (never abstain) with fallback=True if none."""
    ok = [r for r in curve if r["kept_pct"] >= min_coverage_pct]
    if not ok:
        return {"threshold": 0.0, "kept_pct_at_threshold": 100.0, "fallback": True}
    best = max(ok, key=lambda r: r["threshold"])
    return {"threshold": best["threshold"], "kept_pct_at_threshold": best["kept_pct"], "fallback": False}


def evaluate_on_val(model, cols, va, cfg):
    """VAL metrics with 95% bootstrap CIs (challenge F1, AF sensitivity) and the confusion matrix."""
    y_pred, proba = predict(model, cols, va)
    s = summarize(va["label"], y_pred, proba[:, 1])
    for metric in ("challenge_F1", "AF_sensitivity"):
        s[f"{metric}_95CI"] = bootstrap_ci(va["label"], y_pred, metric, n_boot=cfg["n_boot"], seed=cfg["seed"])
    s["confusion"] = confusion(va["label"], y_pred).tolist()
    return s, y_pred, proba


def to_json(obj):
    """Make an object JSON-safe: NaN/inf -> None, numpy numbers -> Python numbers, tuples -> lists."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def val_table(summary):
    """Markdown table of the VAL results (every number comes from summary)."""
    def f(x):
        return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.3f}"

    head = ("| Model | Challenge F1 (95% CI) | F1 N | F1 A | F1 O | F1 ~ | AF sens (95% CI) | AF spec | AF PPV "
            "| AF AUROC |\n|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for name, s in summary["models"].items():
        lo, hi = s["challenge_F1_95CI"]
        slo, shi = s["AF_sensitivity_95CI"]
        rows.append(f"| {name} | {f(s['challenge_F1'])} ({f(lo)}-{f(hi)}) | {f(s['F1_N'])} | {f(s['F1_A'])} "
                    f"| {f(s['F1_O'])} | {f(s['F1_~'])} | {f(s['AF_sensitivity'])} ({f(slo)}-{f(shi)}) "
                    f"| {f(s['AF_specificity'])} | {f(s['AF_PPV'])} | {f(s['AF_AUROC'])} |")
    run = summary["run"]
    return (f"# VAL results (n = {run['n_val']}; models fit on TRAIN, n = {run['n_train']})\n\n" + head + "\n"
            + "\n".join(rows) + "\n")


def main(argv=None):
    """Fit every configured model on TRAIN, evaluate on VAL, save models and results."""
    ap = argparse.ArgumentParser(description="Train on TRAIN, compare on VAL.")
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)

    tr, va = load_split(cfg, "train"), load_split(cfg, "val")
    models_dir, out = Path(cfg["models_dir"]), Path(cfg["out_dir"]) / "val"
    models_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"run": {"n_train": len(tr), "n_val": len(va), "seed": cfg["seed"], "n_boot": cfg["n_boot"],
                       "models": cfg["models"], "sklearn": sklearn.__version__}, "models": {}}
    for name in cfg["models"]:
        model, cols = make_model(name, cfg["seed"])
        model.fit(tr[cols], tr["label"])
        joblib.dump({"model": model, "cols": cols}, models_dir / f"{name}.joblib")
        s, y_pred, proba = evaluate_on_val(model, cols, va, cfg)
        summary["models"][name] = s
        print(f"{name:10s} challenge F1 {s['challenge_F1']:.3f} "
              f"(95% CI {s['challenge_F1_95CI'][0]:.3f}-{s['challenge_F1_95CI'][1]:.3f})  "
              f"AF sens {s['AF_sensitivity']:.3f}  spec {s['AF_specificity']:.3f}")
        if name == "gbm":
            curve = coverage_curve(va["label"], y_pred, proba.max(axis=1), cfg["abstain_grid"])
            choice = choose_abstain_threshold(curve, cfg["abstain_min_coverage_pct"])
            abstain = {**choice, "min_coverage_pct": cfg["abstain_min_coverage_pct"], "chosen_on": "val",
                       "curve": curve}
            (models_dir / "abstain.json").write_text(json.dumps(to_json(abstain), indent=2))
            summary["abstain"] = {k: v for k, v in abstain.items() if k != "curve"}
            if choice["fallback"]:
                print("WARNING: no threshold keeps enough VAL recordings; the app will never abstain.")
            print(f"abstain threshold {choice['threshold']:.2f} keeps {choice['kept_pct_at_threshold']:.1f}% of VAL")

    summary = to_json(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "VAL_table.md").write_text(val_table(summary))
    print(f"Saved {out / 'summary.json'}, {out / 'VAL_table.md'} and models in {models_dir}/")
    return summary


if __name__ == "__main__":
    main()
