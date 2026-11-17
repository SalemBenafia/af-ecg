"""CLI: evaluate frozen models on the test split or an external features CSV (run once, after the freeze).

Usage: python -m afdetect.evaluate --config configs/train.yaml --split test
       python -m afdetect.evaluate --config configs/train.yaml --external data/features/afdb.csv
Refuses to run unless the git tag v1-frozen exists, is an ancestor of HEAD, and models/, configs/ and src/
are unchanged since the tag. Writes results/<name>/summary.json, RESULTS_table.md, confusion_gbm.png and
predictions.csv. External data are bootstrapped by source_record (windows of one person stay together).
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import LABELS  # noqa: E402
from .features import FEATURES  # noqa: E402
from .metrics import bootstrap_ci, confusion, summarize  # noqa: E402
from .models import predict  # noqa: E402
from .train import load_config, labelled_features, to_json  # noqa: E402

FREEZE_TAG = "v1-frozen"
FROZEN_PATHS = ["models", "configs", "src"]
CI_METRICS = ["challenge_F1", "AF_sensitivity", "AF_specificity"]


def _git(*args):
    """Run a git command in the current directory; return (exit code, stdout)."""
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def require_freeze(tag=FREEZE_TAG, paths=FROZEN_PATHS):
    """Stop unless `tag` exists, is an ancestor of HEAD, and `paths` are unchanged since it. Returns the tag commit."""
    code, commit = _git("rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")
    if code != 0:
        raise SystemExit(f"Refusing to evaluate: git tag {tag!r} does not exist. Freeze first (SPEC section 8).")
    if _git("merge-base", "--is-ancestor", tag, "HEAD")[0] != 0:
        raise SystemExit(f"Refusing to evaluate: HEAD does not contain the {tag!r} commit.")
    code, changed = _git("diff", "--name-only", tag, "--", *paths)
    if code != 0 or changed:
        raise SystemExit(f"Refusing to evaluate: files changed since {tag!r}:\n{changed}")
    return commit


def load_test(cfg):
    """Feature rows of the test split (only reached after the freeze check)."""
    df = labelled_features(cfg)
    return df[df["split"] == "test"].reset_index(drop=True)


def load_external(path):
    """External features CSV (e.g. AFDB windows); must have record, source_record, label and all features."""
    df = pd.read_csv(path)
    missing = [c for c in ["record", "source_record", "label"] + FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"External features CSV is missing columns: {missing}")
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels in external CSV: {unknown}")
    return df


def plot_confusion(cm, path, title):
    """Annotated 4x4 confusion matrix (rows = true, columns = predicted)."""
    fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
    ax.imshow(cm, cmap="Greens")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="predicted", ylabel="true", title=title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_model(bundle, df, groups, cfg):
    """Metrics, 95% bootstrap CIs and confusion matrix of one saved model on df."""
    y_pred, proba = predict(bundle["model"], bundle["cols"], df)
    s = summarize(df["label"], y_pred, proba[:, 1])
    for metric in CI_METRICS:
        s[f"{metric}_95CI"] = bootstrap_ci(df["label"], y_pred, metric, groups=groups, n_boot=cfg["n_boot"],
                                           seed=cfg["seed"])
    s["confusion"] = confusion(df["label"], y_pred).tolist()
    return s, y_pred, proba


def abstain_report(df, y_pred, confidence, threshold, groups, cfg):
    """Coverage at the VAL-chosen abstain threshold and the scores on the kept recordings (with CIs)."""
    keep = confidence >= threshold
    out = {"threshold": threshold, "n_kept": int(keep.sum()), "kept_pct": float(100 * keep.mean())}
    if keep.sum() > 10:
        kept_groups = None if groups is None else groups[keep]
        s = summarize(df["label"][keep], y_pred[keep])
        out |= {k: s[k] for k in ["challenge_F1", "AF_sensitivity", "AF_specificity", "AF_PPV", "accuracy"]}
        for metric in ("challenge_F1", "AF_sensitivity"):
            out[f"{metric}_95CI"] = bootstrap_ci(df["label"][keep], y_pred[keep], metric, groups=kept_groups,
                                                 n_boot=cfg["n_boot"], seed=cfg["seed"])
    return out


def results_table(summary):
    """Markdown results table; for external data AF sensitivity/specificity come first."""
    def f(x):
        return "n/a" if x is None else f"{x:.3f}"

    def ci(pair):
        return "n/a" if pair is None or None in pair else f"{pair[0]:.3f}-{pair[1]:.3f}"

    run = summary["run"]
    external = run["kind"] == "external"
    lines = [f"# Results on {run['name']} (n = {run['n']}" + (f", {run['n_groups']} source records" if external
                                                               else "") + ")", ""]
    if external:
        lines += ["Main numbers: AF sensitivity and specificity. AFDB \"(N\" means all non-AF rhythms, not "
                  "\"normal\", so the 4-class challenge F1 is shown for completeness only. CIs resample source "
                  "records.", ""]
    lines += ["| Model | AF sensitivity (95% CI) | AF specificity (95% CI) | AF PPV | AF AUROC | Challenge F1 (95% CI) "
              "| F1 N | F1 A | F1 O | F1 ~ |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary["models"].items():
        lines.append(f"| {name} | {f(s['AF_sensitivity'])} ({ci(s['AF_sensitivity_95CI'])}) "
                     f"| {f(s['AF_specificity'])} ({ci(s['AF_specificity_95CI'])}) | {f(s['AF_PPV'])} "
                     f"| {f(s['AF_AUROC'])} | {f(s['challenge_F1'])} ({ci(s['challenge_F1_95CI'])}) "
                     f"| {f(s['F1_N'])} | {f(s['F1_A'])} | {f(s['F1_O'])} | {f(s['F1_~'])} |")
    ab = summary.get("abstain")
    if ab:
        lines += ["", f"gbm with abstain threshold {ab['threshold']:.2f} (chosen on VAL): keeps {ab['kept_pct']:.1f}% "
                  f"({ab['n_kept']} of {run['n']})" + (f"; on the kept ones challenge F1 {f(ab.get('challenge_F1'))}, "
                                                      f"AF sensitivity {f(ab.get('AF_sensitivity'))}, "
                                                      f"AF specificity {f(ab.get('AF_specificity'))}."
                                                      if "challenge_F1" in ab else ".")]
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Evaluate every configured model once on the test split or an external data set."""
    ap = argparse.ArgumentParser(description="Final evaluation (run once, after the freeze).")
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", choices=["test"])
    g.add_argument("--external", help="features CSV of an external data set (e.g. AFDB windows)")
    a = ap.parse_args(argv)

    freeze_commit = require_freeze()
    cfg = load_config(a.config)
    if a.split:
        df, name, groups, kind = load_test(cfg), "test", None, "split"
    else:
        df, name, kind = load_external(a.external), Path(a.external).stem, "external"
        groups = df["source_record"].astype(str).to_numpy()
    out = Path(cfg["out_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    models_dir = Path(cfg["models_dir"])
    abstain = json.loads((models_dir / "abstain.json").read_text())

    summary = {"run": {"name": name, "kind": kind, "n": len(df), "n_groups": None if groups is None
                       else int(len(np.unique(groups))), "freeze_tag": FREEZE_TAG, "freeze_commit": freeze_commit,
                       "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "n_boot": cfg["n_boot"], "seed": cfg["seed"]}, "models": {}}
    for model_name in cfg["models"]:
        bundle = joblib.load(models_dir / f"{model_name}.joblib")
        s, y_pred, proba = evaluate_model(bundle, df, groups, cfg)
        summary["models"][model_name] = s
        if model_name == "gbm":
            confidence = proba.max(axis=1)
            summary["abstain"] = abstain_report(df, y_pred, confidence, abstain["threshold"], groups, cfg)
            plot_confusion(np.array(s["confusion"]), out / "confusion_gbm.png", f"gbm on {name} (n = {len(df)})")
            preds = pd.DataFrame({"record": df["record"], "label": df["label"], "pred": y_pred,
                                  "p_af": proba[:, 1], "confidence": confidence,
                                  "abstained": confidence < abstain["threshold"]})
            if groups is not None:
                preds.insert(1, "source_record", groups)
            preds.to_csv(out / "predictions.csv", index=False)

    summary = to_json(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    table = results_table(summary)
    (out / "RESULTS_table.md").write_text(table)
    print(table)
    print(f"Saved summary.json, RESULTS_table.md, confusion_gbm.png and predictions.csv in {out}/")
    return summary


if __name__ == "__main__":
    main()
