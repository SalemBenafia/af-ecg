"""Post-freeze error analysis of the gbm on the test split (no retraining, no model changes).

Usage: python scripts/errors.py --predictions results/test/predictions.csv \
           --features data/features/cinc2017.csv --root data/training2017 --out results/errors
Writes subgroups.csv / subgroups.md (challenge F1 and AF sensitivity by length, heart rate and signal quality,
with 95% bootstrap CIs) and 6 error plots: 2 AF missed, 2 false AF alarms, 2 AF called Other.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from afdetect.data import NAMES, read_wfdb  # noqa: E402
from afdetect.features import FS, preprocess  # noqa: E402
from afdetect.metrics import bootstrap_ci, summarize  # noqa: E402

SMALL_N, SMALL_AF = 100, 20   # groups below either size are descriptive only
PLOT_SECONDS = 15
N_BOOT, SEED = 2000, 0  # same bootstrap settings as configs/train.yaml


def load(predictions_csv, features_csv):
    """Test predictions joined with duration_s, hr_mean and nk_quality from the features CSV."""
    preds = pd.read_csv(predictions_csv)
    feats = pd.read_csv(features_csv, usecols=["record", "duration_s", "hr_mean", "nk_quality"])
    df = preds.merge(feats, on="record", how="left", validate="one_to_one")
    assert len(df) == len(preds)
    return df


def add_groups(df):
    """Subgroup columns: length (<20, 20-40, >40 s), heart rate (<60, 60-100, >100 bpm) and quality terciles."""
    out = df.copy()
    edges = out["nk_quality"].quantile([1 / 3, 2 / 3]).round(3).tolist()
    out["length"] = _labelled(pd.cut(out["duration_s"], [0, 20, 40, np.inf], right=False,
                                     labels=["< 20 s", "20-40 s", ">= 40 s"]))
    out["heart_rate"] = _labelled(pd.cut(out["hr_mean"], [0, 60, 100, np.inf], right=False,
                                         labels=["< 60 bpm", "60-100 bpm", ">= 100 bpm"]))
    out["quality"] = _labelled(pd.qcut(out["nk_quality"], 3, labels=[
        f"low (< {edges[0]})", f"middle ({edges[0]}-{edges[1]})", f"high (> {edges[1]})"]))
    return out


def _labelled(categories):
    """Categorical bins as plain strings, with missing values (NaN features) as "unknown (NaN)"."""
    return categories.cat.add_categories("unknown (NaN)").fillna("unknown (NaN)").astype(str)


GROUP_ORDER = ["< 20 s", "20-40 s", ">= 40 s", "< 60 bpm", "60-100 bpm", ">= 100 bpm", "low", "middle", "high",
               "unknown"]


def _rank(group):
    """Position of a group label in GROUP_ORDER (quality labels match on their first word)."""
    return next(i for i, g in enumerate(GROUP_ORDER) if group == g or group.split(" ")[0] == g)


def subgroup_table(df, columns=("length", "heart_rate", "quality"), n_boot=N_BOOT):
    """One row per subgroup: n, n_AF, challenge F1 and AF sensitivity with 95% bootstrap CIs."""
    rows = []
    for col in columns:
        for group in sorted(df[col].unique(), key=_rank):
            part = df[df[col] == group]
            s = summarize(part["label"], part["pred"])
            n_af = int((part["label"] == "A").sum())
            f1_ci = bootstrap_ci(part["label"], part["pred"], "challenge_F1", n_boot=n_boot, seed=SEED)
            sens_ci = (bootstrap_ci(part["label"], part["pred"], "AF_sensitivity", n_boot=n_boot, seed=SEED)
                       if n_af else (np.nan, np.nan))
            rows.append({"by": col, "group": group, "n": len(part), "n_AF": n_af,
                         "challenge_F1": s["challenge_F1"], "F1_lo": f1_ci[0], "F1_hi": f1_ci[1],
                         "AF_sensitivity": s["AF_sensitivity"], "sens_lo": sens_ci[0], "sens_hi": sens_ci[1],
                         "descriptive_only": len(part) < SMALL_N or n_af < SMALL_AF})
    return pd.DataFrame(rows)


def table_markdown(table):
    """Markdown version of the subgroup table; small groups are flagged."""
    def f(x):
        return "n/a" if pd.isna(x) else f"{x:.3f}"

    lines = ["| By | Group | n | n AF | Challenge F1 (95% CI) | AF sensitivity (95% CI) | Note |",
             "|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        note = "descriptive only (small group)" if r.descriptive_only else ""
        lines.append(f"| {r.by} | {r.group} | {r.n} | {r.n_AF} | {f(r.challenge_F1)} ({f(r.F1_lo)}-{f(r.F1_hi)}) "
                     f"| {f(r.AF_sensitivity)} ({f(r.sens_lo)}-{f(r.sens_hi)}) | {note} |")
    return "\n".join(lines) + "\n"


def pick_errors(df, per_kind=2):
    """The most confident errors of three kinds (deterministic: sorted by p_af, then record)."""
    af_missed = df[(df["label"] == "A") & df["pred"].isin(["N", "~"])].sort_values(["p_af", "record"])
    false_alarm = df[(df["label"] != "A") & (df["pred"] == "A")].sort_values(["p_af", "record"],
                                                                             ascending=[False, True])
    af_as_other = df[(df["label"] == "A") & (df["pred"] == "O")].sort_values(["p_af", "record"])
    kinds = {"af_missed": af_missed, "false_af_alarm": false_alarm, "af_called_other": af_as_other}
    return [(kind, row) for kind, part in kinds.items() for _, row in part.head(per_kind).iterrows()]


def plot_error(root, kind, row, out_png, seconds=PLOT_SECONDS):
    """Cleaned ECG (first `seconds`) with R peaks, and the RR intervals of the whole recording."""
    x, fs = read_wfdb(Path(root) / row["record"])
    _, cleaned, rpeaks = preprocess(x, fs)
    n = min(len(cleaned), seconds * FS)
    shown = rpeaks[rpeaks < n]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.2), gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(np.arange(n) / FS, cleaned[:n], lw=0.7, color="#1f4e79")
    ax1.plot(shown / FS, cleaned[shown], "o", ms=4, color="#d62728")
    ax1.set(xlim=(0, seconds), xlabel="time (s)", ylabel="mV (cleaned)")
    ax1.set_title(f"{kind.replace('_', ' ')} · {row['record']} · true {NAMES[row['label']]} → predicted "
                  f"{NAMES[row['pred']]} · p(AF) {row['p_af']:.2f} · confidence {row['confidence']:.2f} · "
                  f"HR {row['hr_mean']:.0f} bpm · quality {row['nk_quality']:.2f}", fontsize=9, loc="left")
    rr = np.diff(rpeaks) / FS * 1000
    ax2.plot(np.arange(1, len(rr) + 1), rr, "o-", ms=3, lw=0.8, color="#7c3aed")
    ax2.set(xlabel="beat number (whole recording)", ylabel="RR (ms)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main(argv=None):
    """Subgroup tables and error plots."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", default="results/test/predictions.csv")
    ap.add_argument("--features", default="data/features/cinc2017.csv")
    ap.add_argument("--root", default="data/training2017")
    ap.add_argument("--out", default="results/errors")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = add_groups(load(a.predictions, a.features))
    table = subgroup_table(df, n_boot=a.n_boot)
    table.to_csv(out / "subgroups.csv", index=False)
    (out / "subgroups.md").write_text(table_markdown(table))
    print(table_markdown(table))

    picks = pick_errors(df)
    for i, (kind, row) in enumerate(picks, 1):
        png = out / f"{i}_{kind}_{row['record']}.png"
        plot_error(a.root, kind, row, png)
        print(f"{png.name}: true {row['label']} pred {row['pred']} p_af {row['p_af']:.3f} "
              f"conf {row['confidence']:.3f} HR {row['hr_mean']:.1f} quality {row['nk_quality']:.3f} "
              f"length {row['duration_s']:.1f} s")
    return table, picks


if __name__ == "__main__":
    main()
