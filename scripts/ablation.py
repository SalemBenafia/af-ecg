"""gbm ablation on VAL: feature sets x class_weight x learning_rate, several seeds, plus permutation importance.

Usage: python scripts/ablation.py --config configs/train.yaml --out results/val --seeds 0 1 2
Fits on TRAIN, scores on VAL only (train.load_split refuses the test split). Writes ablation.csv (one row per
run), permutation_importance.csv and ablation.md. A hand-written "## Recommendation" section already in
ablation.md is kept when the script is re-run. configs/train.yaml is never modified.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from afdetect.features import FEATURES, RR_FEATURES
from afdetect.metrics import summarize
from afdetect.models import make_model, predict
from afdetect.train import load_config, load_split

FEATURE_SETS = {
    "RR only": list(RR_FEATURES),
    "RR + P wave": list(RR_FEATURES) + ["p_wave_ratio"],
    "RR + P wave + f-wave": list(RR_FEATURES) + ["p_wave_ratio", "fwave_ratio"],
    "all features": list(FEATURES),
}
CLASS_WEIGHTS = ["balanced", None]
LEARNING_RATES = [0.05, 0.1]
RECOMMENDATION_HEADER = "## Recommendation"


def fit_gbm(tr, cols, class_weight, learning_rate, seed):
    """The configured gbm with two settings changed, fit on TRAIN columns `cols`."""
    model, _ = make_model("gbm", seed)
    model.set_params(class_weight=class_weight, learning_rate=learning_rate)
    return model.fit(tr[cols], tr["label"])


def run_grid(tr, va, seeds):
    """One row per (feature set, class_weight, learning_rate, seed) with VAL challenge F1 and AF sensitivity."""
    rows = []
    for (set_name, cols), cw, lr, seed in itertools.product(FEATURE_SETS.items(), CLASS_WEIGHTS, LEARNING_RATES,
                                                            seeds):
        model = fit_gbm(tr, cols, cw, lr, seed)
        y_pred, proba = predict(model, cols, va)
        s = summarize(va["label"], y_pred, proba[:, 1])
        rows.append({"feature_set": set_name, "n_features": len(cols), "class_weight": str(cw),
                     "learning_rate": lr, "seed": seed, "challenge_F1": s["challenge_F1"],
                     "AF_sensitivity": s["AF_sensitivity"], "AF_specificity": s["AF_specificity"],
                     "n_iter": int(model.n_iter_)})
    return pd.DataFrame(rows)


def aggregate(runs):
    """Mean and SD over seeds per configuration, best mean challenge F1 first."""
    keys = ["feature_set", "n_features", "class_weight", "learning_rate"]
    agg = runs.groupby(keys, sort=False).agg(
        F1_mean=("challenge_F1", "mean"), F1_std=("challenge_F1", "std"),
        sens_mean=("AF_sensitivity", "mean"), sens_std=("AF_sensitivity", "std"),
        spec_mean=("AF_specificity", "mean"), n_seeds=("seed", "count")).reset_index()
    return agg.sort_values("F1_mean", ascending=False, ignore_index=True)


def challenge_f1_scorer(estimator, X, y):
    """Scorer for permutation_importance: VAL challenge F1 of the estimator's predictions."""
    return summarize(y, estimator.predict(X))["challenge_F1"]


def importance(tr, va, best, seed, n_repeats=10):
    """Permutation importance (drop in VAL challenge F1) of the best configuration, fit with `seed`."""
    cols = FEATURE_SETS[best["feature_set"]]
    cw = None if best["class_weight"] == "None" else best["class_weight"]
    model = fit_gbm(tr, cols, cw, best["learning_rate"], seed)
    r = permutation_importance(model, va[cols], va["label"], scoring=challenge_f1_scorer,
                               n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return pd.DataFrame({"feature": cols, "importance_mean": r.importances_mean,
                         "importance_std": r.importances_std}).sort_values(
        "importance_mean", ascending=False, ignore_index=True)


def markdown(agg, imp, run_info, recommendation):
    """ablation.md: the configuration table, permutation importance and the recommendation section."""
    lines = [f"# gbm ablation on VAL (fit on TRAIN n = {run_info['n_train']}, scored on VAL n = {run_info['n_val']})",
             "", f"Seeds: {run_info['seeds']}. Values are mean ± SD over seeds. The configured model is "
             "*all features, balanced, 0.05*. The test split was not used.", "",
             "| Feature set | # features | class_weight | learning_rate | Challenge F1 | AF sensitivity "
             "| AF specificity (mean) |", "|---|---|---|---|---|---|---|"]
    for r in agg.itertuples():
        lines.append(f"| {r.feature_set} | {r.n_features} | {r.class_weight} | {r.learning_rate} "
                     f"| {r.F1_mean:.3f} ± {r.F1_std:.3f} | {r.sens_mean:.3f} ± {r.sens_std:.3f} | {r.spec_mean:.3f} |")
    best = agg.iloc[0]
    lines += ["", f"## Permutation importance (best configuration: {best.feature_set}, class_weight "
              f"{best.class_weight}, learning_rate {best.learning_rate}; seed {run_info['seeds'][0]}, "
              f"{run_info['n_repeats']} repeats on VAL)", "",
              "Drop in VAL challenge F1 when one feature is shuffled. Correlated features share credit, so a "
              "low value does not mean the feature is useless.", "", "| Feature | Mean drop | SD |", "|---|---|---|"]
    lines += [f"| {r.feature} | {r.importance_mean:.4f} | {r.importance_std:.4f} |" for r in imp.itertuples()]
    lines += ["", recommendation or f"{RECOMMENDATION_HEADER}\n\n(to be written after reading the table)\n"]
    return "\n".join(lines).rstrip() + "\n"


def existing_recommendation(md_path):
    """The hand-written recommendation section of an existing ablation.md, or None."""
    if not Path(md_path).exists():
        return None
    text = Path(md_path).read_text()
    i = text.find(RECOMMENDATION_HEADER)
    return text[i:] if i >= 0 else None


def main(argv=None):
    """Run the grid, aggregate, compute importance and write the three output files."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--out", default="results/val")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-repeats", type=int, default=10)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    tr, va = load_split(cfg, "train"), load_split(cfg, "val")
    runs = run_grid(tr, va, a.seeds)
    agg = aggregate(runs)
    imp = importance(tr, va, agg.iloc[0], a.seeds[0], a.n_repeats)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "ablation.csv", index=False)
    imp.to_csv(out / "permutation_importance.csv", index=False)
    info = {"n_train": len(tr), "n_val": len(va), "seeds": a.seeds, "n_repeats": a.n_repeats}
    md_path = out / "ablation.md"
    md_path.write_text(markdown(agg, imp, info, existing_recommendation(md_path)))
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(agg.to_string(index=False))
        print("\nPermutation importance:\n" + imp.to_string(index=False))
    print(f"Saved {out / 'ablation.csv'}, {out / 'permutation_importance.csv'}, {md_path}")
    return agg, imp


if __name__ == "__main__":
    main()
