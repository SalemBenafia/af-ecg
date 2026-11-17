"""Train all models on TRAIN, compare them on VAL, choose the abstain threshold on VAL.
    python -m afdetect.train --config configs/train.yaml
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .metrics import bootstrap_ci, coverage_curve, summarize
from .models import make_model, predict


def load_split(cfg, split):
    feats = pd.read_csv(cfg["features_csv"])
    splits = pd.read_csv(cfg["split_file"])
    df = feats.merge(splits[["record", "split"]], on="record")
    return df[df["split"] == split].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tr, va = load_split(cfg, "train"), load_split(cfg, "val")
    Path(cfg["models_dir"]).mkdir(parents=True, exist_ok=True)
    out = Path(cfg["out_dir"]) / "val"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in cfg["models"]:
        model, cols = make_model(name, cfg["seed"])
        model.fit(tr[cols], tr["label"])
        joblib.dump({"model": model, "cols": cols}, Path(cfg["models_dir"]) / f"{name}.joblib")
        y_pred, proba = predict(model, cols, va)
        s = summarize(va["label"], y_pred, proba[:, 1])
        s["challenge_F1_95CI"] = bootstrap_ci(va["label"], y_pred, n_boot=cfg["n_boot"], seed=cfg["seed"])
        summary[name] = s
        print(f"{name:10s} challenge F1={s['challenge_F1']:.3f} "
              f"(95% CI {s['challenge_F1_95CI'][0]:.3f}-{s['challenge_F1_95CI'][1]:.3f})  "
              f"AF sens={s['AF_sensitivity']:.3f} spec={s['AF_specificity']:.3f}")
        if name == "gbm":                    # abstain threshold, chosen on VAL only
            conf = proba.max(1)
            curve = coverage_curve(va["label"], y_pred, conf, np.arange(0.30, 0.96, 0.05))
            ok = [r for r in curve if r["kept_pct"] >= cfg["abstain_min_coverage_pct"]]
            th = max(r["threshold"] for r in ok) if ok else 0.0
            json.dump({"threshold": th, "curve": curve}, open(Path(cfg["models_dir"]) / "abstain.json", "w"),
                      indent=2)
            print(f"abstain threshold (keeps >= {cfg['abstain_min_coverage_pct']}% of val): {th:.2f}")
    json.dump(summary, open(out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
