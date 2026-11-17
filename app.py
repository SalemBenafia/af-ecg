"""AF Detect demo (Gradio): upload a short single-lead ECG or pick an example -> rhythm, confidence, plots.

Run: python app.py. Uses the frozen models/gbm.joblib and models/abstain.json. Set AFDETECT_HOME to load models,
examples and reference values from another folder (used by the tests).
"""
import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "src"))  # works without `pip install -e .` (e.g. on a Hugging Face Space)

try:  # Hugging Face ZeroGPU: installed on the Space (deploy/space/requirements.txt), absent locally
    import spaces  # noqa: E402
except ImportError:
    spaces = None

import gradio as gr  # noqa: E402
import joblib  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from afdetect.data import LABELS, NAMES, read_csv_ecg, read_uploaded  # noqa: E402
from afdetect.features import FS, extract_features, preprocess  # noqa: E402
from afdetect.models import predict  # noqa: E402

BASE = Path(os.environ.get("AFDETECT_HOME", APP_DIR))
MIN_SECONDS = 9
BUNDLE = joblib.load(BASE / "models" / "gbm.joblib")
THRESHOLD = json.loads((BASE / "models" / "abstain.json").read_text())["threshold"]
EXAMPLES = {p.stem: p for p in sorted((BASE / "examples").glob("*.csv"))}
DISCLAIMER = ("**Research prototype from a student hackathon. Not a medical device. It cannot rule out heart "
              "disease. If you have symptoms, contact a doctor or emergency services.**")
WHY_FEATURES = [("rmssd_norm", "beat-to-beat variability"), ("p_wave_ratio", "P-wave strength"),
                ("tmpl_corr", "beat-shape consistency")]


def load_reference_medians(path=BASE / "results" / "explore" / "feature_medians_train.csv"):
    """TRAIN medians per class (rows = features), used as reference values in the "Why" line; None if missing."""
    path = Path(path)
    return pd.read_csv(path, index_col=0) if path.exists() else None


MEDIANS = load_reference_medians()


if spaces is not None:
    @spaces.GPU
    def zerogpu_placeholder():
        """Never called. ZeroGPU Spaces need at least one @spaces.GPU function to start; the model runs on CPU."""
        return None


def fmt(x, digits=2):
    """A number for display, or "not measurable" for NaN."""
    return f"{x:.{digits}f}" if np.isfinite(x) else "not measurable"


def load_input(files, csv_fs, example):
    """The ECG to analyse: the chosen example first, otherwise the uploaded file(s)."""
    if example and example != "(none)":
        return read_csv_ecg(EXAMPLES[example], FS)
    if files:
        paths = [f if isinstance(f, (str, Path)) else f.name for f in files]
        try:
            return read_uploaded(paths, csv_fs)
        except Exception as e:
            raise gr.Error(f"Could not read the file: {e}")
    raise gr.Error("Upload a recording or pick an example first.")


def why_line(f):
    """Plain-language explanation: three key features next to typical TRAIN values for Normal and AF."""
    parts = []
    for key, name in WHY_FEATURES:
        ref = ""
        if MEDIANS is not None and key in MEDIANS.index:
            ref = f" (typical Normal {MEDIANS.loc[key, 'N']:.2f}, typical AF {MEDIANS.loc[key, 'A']:.2f})"
        parts.append(f"{name} = {fmt(f[key])}{ref}")
    return "**Why:** " + "; ".join(parts) + ". Typical values are medians of the training recordings."


def verdict(label, proba, f):
    """Markdown verdict: the class (or Uncertain below the abstain threshold), advice, heart rate and why."""
    conf = float(proba.max())
    hr = f"{fmt(f['hr_mean'], 0)} bpm" if np.isfinite(f["hr_mean"]) else "not measurable"
    if conf < THRESHOLD:
        head = (f"## Uncertain\nThe model is not confident enough (top probability {conf:.0%}, below "
                f"{THRESHOLD:.0%}), so it gives no answer. A clinician should review this recording.")
    elif label == "A":
        head = (f"## {NAMES[label]}\nConfidence {conf:.0%}. This looks like atrial fibrillation: a clinician "
                "should review this recording.")
    elif label == "~":
        head = f"## {NAMES[label]}\nConfidence {conf:.0%}. The signal is too noisy to judge: please record again."
    else:
        head = f"## {NAMES[label]}\nConfidence {conf:.0%}."
    return f"{head}\n\nAverage heart rate: {hr}.\n\n{why_line(f)}\n\n_Screening aid only, not a diagnosis._"


def make_figure(ecg, fs):
    """Three panels: 10 s of cleaned ECG with R peaks, RR intervals by beat, Poincare plot."""
    _, cleaned, rpeaks = preprocess(ecg, fs)
    rr = np.diff(rpeaks) / FS * 1000
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    n = min(len(cleaned), 10 * FS)
    shown = rpeaks[rpeaks < n]
    ax[0].plot(np.arange(n) / FS, cleaned[:n], color="#0F766E", lw=0.8)
    ax[0].plot(shown / FS, cleaned[shown], "o", color="#B45309", ms=4)
    ax[0].set(title="ECG, first 10 s (dots = detected heartbeats)", xlabel="time (s)", ylabel="mV")
    if len(rr) > 1:
        ax[1].plot(np.arange(1, len(rr) + 1), rr, marker="o", ms=3, color="#0F766E")
        ax[2].plot(rr[:-1], rr[1:], "o", ms=4, alpha=0.7, color="#7C3AED")
    ax[1].set(title="Time between beats (irregular jumps suggest AF)", xlabel="beat number", ylabel="ms")
    ax[2].set(title="Poincaré plot: tight cluster = regular, wide cloud = irregular",
              xlabel="interval n (ms)", ylabel="interval n+1 (ms)")
    for a in ax:
        a.grid(alpha=0.3)
    return fig


def analyze(files, csv_fs, example):
    """Read, check length, extract features, predict; return (probabilities, verdict markdown, figure)."""
    ecg, fs = load_input(files, csv_fs, example)
    if len(ecg) / fs < MIN_SECONDS:
        raise gr.Error(f"Recording too short: at least {MIN_SECONDS} seconds are needed "
                       f"(this one is {len(ecg) / fs:.1f} s).")
    f = extract_features(ecg, fs)
    labels, proba = predict(BUNDLE["model"], BUNDLE["cols"], pd.DataFrame([f]))
    probs = {NAMES[c]: float(p) for c, p in zip(LABELS, proba[0])}
    return probs, verdict(labels[0], proba[0], f), make_figure(ecg, fs)


with gr.Blocks(title="AF Detect: ECG rhythm check") as demo:
    gr.Markdown("# AF Detect: is this heart rhythm atrial fibrillation?\n\n" + DISCLAIMER + "\n\n"
                "Upload a short single-lead ECG (9-61 s) or pick an example from the held-out test data. The model "
                "answers Normal, Atrial fibrillation, Other rhythm, Too noisy, or Uncertain when it is not confident.")
    with gr.Row():
        with gr.Column():
            files = gr.File(label="Upload a .csv (one column) or a .hea + .mat/.dat pair", file_count="multiple",
                            file_types=[".csv", ".hea", ".mat", ".dat"])
            csv_fs = gr.Number(value=FS, label="Sampling rate of the CSV file (Hz)")
            example = gr.Dropdown(["(none)"] + list(EXAMPLES), value="(none)", label="Or pick an example")
            button = gr.Button("Analyze", variant="primary")
        with gr.Column():
            probabilities = gr.Label(num_top_classes=4, label="Probabilities")
            result = gr.Markdown()
    plot = gr.Plot(label="What the model looked at")
    button.click(analyze, inputs=[files, csv_fs, example], outputs=[probabilities, result, plot], api_name="analyze")

if __name__ == "__main__":
    demo.launch()
