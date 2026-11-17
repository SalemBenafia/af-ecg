"""Box plot of every feature by class, and the median per class, on the TRAIN split only.

Usage: python scripts/feature_plots.py --features data/features/cinc2017.csv --splits configs/splits.csv \
           --out results/explore
Writes results/explore/features_by_class.png, feature_medians_train.csv and feature_auc_train.csv
(single-feature AUROC for AF vs Normal and AF vs Other).
Only TRAIN records are used: VAL and TEST stay unseen for model decisions.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from afdetect.data import LABELS, NAMES  # noqa: E402
from afdetect.features import FEATURES  # noqa: E402

COLORS = {"N": "#2b8cbe", "A": "#e34a33", "O": "#31a354", "~": "#969696"}
N_COLS = 4


def load_train(features_csv, splits_csv):
    """Feature rows of TRAIN records only (the split file decides, not the features file)."""
    feats = pd.read_csv(features_csv)
    splits = pd.read_csv(splits_csv, dtype=str)
    train = set(splits.loc[splits["split"] == "train", "record"])
    return feats[feats["record"].isin(train)].reset_index(drop=True)


def median_table(df):
    """Median of every feature per class (rows = features, columns = N, A, O, ~), plus n per class."""
    table = df.groupby("label")[FEATURES].median().reindex(LABELS).T
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    return pd.concat([pd.DataFrame([counts.to_numpy()], index=["n_records"], columns=LABELS), table])


def single_feature_auc(df, positive="A", negative="N"):
    """AUROC of each feature alone for `positive` vs `negative` (NaN rows dropped per feature).

    auc > 0.5: the feature is higher in `positive`; separation = |auc - 0.5| * 2 (0 = none, 1 = perfect)."""
    pair = df[df["label"].isin([positive, negative])]
    rows = []
    for feat in FEATURES:
        sub = pair[["label", feat]].dropna()
        y = (sub["label"] == positive).to_numpy()
        auc = roc_auc_score(y, sub[feat]) if 0 < y.sum() < len(y) and sub[feat].nunique() > 1 else np.nan
        rows.append({"feature": feat, "auc": auc, "separation": abs(auc - 0.5) * 2,
                     "higher_in": positive if auc > 0.5 else negative, "n_used": len(sub)})
    return pd.DataFrame(rows).sort_values("separation", ascending=False, ignore_index=True)


def plot_boxes(df, out_png):
    """One box plot per feature, one box per class; outliers hidden so the boxes stay readable."""
    n_rows = int(np.ceil(len(FEATURES) / N_COLS))
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(4.2 * N_COLS, 2.8 * n_rows))
    for ax, feat in zip(axes.flat, FEATURES):
        data = [df.loc[df["label"] == c, feat].dropna().to_numpy() for c in LABELS]
        box = ax.boxplot(data, tick_labels=LABELS, showfliers=False, patch_artist=True, widths=0.6)
        for patch, c in zip(box["boxes"], LABELS):
            patch.set_facecolor(COLORS[c])
            patch.set_alpha(0.6)
        ax.set_title(feat, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    for ax in list(axes.flat)[len(FEATURES):]:
        ax.axis("off")
    legend = ", ".join(f"{c} = {NAMES[c]}" for c in LABELS)
    fig.suptitle(f"Features by class, TRAIN split only (n = {len(df)}; outliers hidden)\n{legend}", fontsize=11)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main(argv=None):
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default="data/features/cinc2017.csv")
    ap.add_argument("--splits", default="configs/splits.csv")
    ap.add_argument("--out", default="results/explore")
    a = ap.parse_args(argv)

    df = load_train(a.features, a.splits)
    out = Path(a.out)
    plot_boxes(df, out / "features_by_class.png")
    table = median_table(df)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "feature_medians_train.csv")
    aucs = pd.concat([single_feature_auc(df, "A", "N").assign(comparison="A_vs_N"),
                      single_feature_auc(df, "A", "O").assign(comparison="A_vs_O")], ignore_index=True)
    aucs.to_csv(out / "feature_auc_train.csv", index=False)
    with pd.option_context("display.width", 120, "display.float_format", "{:.4g}".format):
        print(table.to_string())
        for comp, part in aucs.groupby("comparison", sort=False):
            print(f"\nTop 5 single features, {comp} (TRAIN):")
            print(part.head(5).drop(columns="comparison").to_string(index=False))
    print(f"\nSaved features_by_class.png, feature_medians_train.csv, feature_auc_train.csv in {out} "
          f"(TRAIN, n = {len(df)})")
    return table


if __name__ == "__main__":
    main()
