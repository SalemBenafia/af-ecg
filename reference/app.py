"""Gradio demo: upload a single-lead ECG (or pick an example) -> rhythm class, confidence, plots."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))   # works without `pip install -e .`

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

BUNDLE = joblib.load("models/gbm.joblib")
ABSTAIN = json.load(open("models/abstain.json"))["threshold"]
EXAMPLES = {p.stem: p for p in sorted(Path("examples").glob("*.csv"))}   # 300 Hz, from the TEST split
DISCLAIMER = ("**Research prototype from a student hackathon. Not a medical device.** It cannot rule out "
              "heart disease. If you have symptoms, contact a doctor or emergency services.")


def make_figure(ecg, fs):
    _, cleaned, rpeaks = preprocess(ecg, fs)
    t = np.arange(len(cleaned)) / FS
    rr = np.diff(rpeaks) / FS
    fig, ax = plt.subplots(3, 1, figsize=(7, 7.5), constrained_layout=True)
    n = min(len(cleaned), 10 * FS)
    ax[0].plot(t[:n], cleaned[:n], color="#0F766E", lw=0.8)
    pk = rpeaks[rpeaks < n]
    ax[0].plot(t[pk], cleaned[pk], "o", color="#B45309", ms=4)
    ax[0].set(title="ECG (first 10 s), dots = detected heartbeats", xlabel="time (s)", yticks=[])
    if len(rr) > 1:
        ax[1].plot(rr * 1000, marker="o", ms=3, color="#0F766E")
        ax[2].plot(rr[:-1] * 1000, rr[1:] * 1000, "o", ms=4, alpha=0.7, color="#7C3AED")
    ax[1].set(title="Time between beats (irregular jumps suggest AF)", xlabel="beat number", ylabel="ms")
    ax[2].set(title="Poincaré plot: tight cluster = regular, wide cloud = irregular",
              xlabel="interval n (ms)", ylabel="interval n+1 (ms)")
    return fig


def analyze(files, csv_fs, example):
    if example and example != "(none)":
        ecg, fs = read_csv_ecg(EXAMPLES[example], 300)
    elif files:
        paths = [f if isinstance(f, str) else f.name for f in files]
        try:
            ecg, fs = read_uploaded(paths, csv_fs)
        except Exception as e:
            raise gr.Error(f"Could not read the file: {e}")
    else:
        raise gr.Error("Upload a recording or pick an example first.")
    if len(ecg) / fs < 9:
        raise gr.Error("Recording too short: at least 9 seconds are needed.")
    f = extract_features(ecg, fs)
    label, proba = predict(BUNDLE["model"], BUNDLE["cols"], pd.DataFrame([f]))
    label, proba = label[0], proba[0]
    conf = float(proba.max())
    probs = {NAMES[c]: float(p) for c, p in zip(LABELS, proba)}
    hr = f"{f['hr_mean']:.0f} bpm" if np.isfinite(f["hr_mean"]) else "not measurable"
    if conf < ABSTAIN:
        text = (f"## Uncertain\nThe model is not confident enough (top probability {conf:.0%}), so it gives "
                f"no answer. A clinician should review this recording. Average heart rate: {hr}.")
    else:
        text = f"## {NAMES[label]}\nConfidence {conf:.0%}  ·  average heart rate {hr}."
    text += (f"\n\n**Why:** beat-to-beat variability (normalised RMSSD) = {f['rmssd_norm']:.2f}, "
             f"P-wave strength = {f['p_wave_ratio']:.2f}, beat-shape consistency = {f['tmpl_corr']:.2f}.")
    return probs, text, make_figure(ecg, fs)


with gr.Blocks(title="ECG rhythm check (AF)") as demo:
    gr.Markdown("# ECG rhythm check: atrial fibrillation\n" + DISCLAIMER)
    with gr.Row():
        with gr.Column():
            files = gr.File(label="Upload a .csv (one column) or a .hea + .mat/.dat pair", file_count="multiple",
                            file_types=[".csv", ".hea", ".mat", ".dat"])
            csv_fs = gr.Number(value=300, label="Sampling rate of the CSV file (Hz)")
            example = gr.Dropdown(["(none)"] + list(EXAMPLES), value="(none)", label="Or pick an example")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            probs = gr.Label(num_top_classes=4, label="Probabilities")
            result = gr.Markdown()
    plot = gr.Plot(label="What the model looked at")
    btn.click(analyze, inputs=[files, csv_fs, example], outputs=[probs, result, plot])

if __name__ == "__main__":
    demo.launch()
