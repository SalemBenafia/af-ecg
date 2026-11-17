"""Final evaluation, run ONCE per data set after all choices are frozen.
    python -m afdetect.evaluate --config configs/train.yaml --split test
    python -m afdetect.evaluate --config configs/train.yaml --external data/features/afdb.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from .data import LABELS  # noqa: E402
from .metrics import bootstrap_ci, confusion, coverage_curve, summarize  # noqa: E402
from .models import predict  # noqa: E402
from .train import load_split  # noqa: E402


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(4.2, 3.8), constrained_layout=True)
    ax.imshow(cm, cmap="Greens")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="predicted", ylabel="true", title=title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", choices=["test"])
    g.add_argument("--external", help="features CSV of an external data set (e.g. AFDB windows)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.split:
        df, name, groups = load_split(cfg, "test"), "test", None
    else:
        df = pd.read_csv(a.external)
        name, groups = Path(a.external).stem, df["source_record"].to_numpy()
    out = Path(cfg["out_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    abstain = json.load(open(Path(cfg["models_dir"]) / "abstain.json"))["threshold"]
    summary, lines = {}, ["| Model | Challenge F1 (95% CI) | AF sensitivity (95% CI) | AF specificity | AF PPV |",
                          "|---|---|---|---|---|"]
    for m in cfg["models"]:
        bundle = joblib.load(Path(cfg["models_dir"]) / f"{m}.joblib")
        y_pred, proba = predict(bundle["model"], bundle["cols"], df)
        s = summarize(df["label"], y_pred, proba[:, 1])
        nb, seed = cfg["n_boot"], cfg["seed"]
        s["challenge_F1_95CI"] = bootstrap_ci(df["label"], y_pred, "challenge_F1", groups, nb, seed)
        s["AF_sensitivity_95CI"] = bootstrap_ci(df["label"], y_pred, "AF_sensitivity", groups, nb, seed)
        s["confusion"] = confusion(df["label"], y_pred).tolist()
        if m == "gbm":
            s["abstain_threshold"] = abstain
            s["coverage"] = coverage_curve(df["label"], y_pred, proba.max(1), [0.0, abstain])
            plot_confusion(np.array(s["confusion"]), out / "confusion_gbm.png", f"GBM on {name}")
            pd.DataFrame({"record": df["record"], "label": df["label"], "pred": y_pred,
                          "p_af": proba[:, 1], "confidence": proba.max(1)}).to_csv(out / "predictions.csv",
                                                                                    index=False)
        summary[m] = s
        ci, sc = s["challenge_F1_95CI"], s["AF_sensitivity_95CI"]
        lines.append(f"| {m} | {s['challenge_F1']:.3f} ({ci[0]:.3f}-{ci[1]:.3f}) | {s['AF_sensitivity']:.3f} "
                     f"({sc[0]:.3f}-{sc[1]:.3f}) | {s['AF_specificity']:.3f} | {s['AF_PPV']:.3f} |")
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    (out / "RESULTS_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
