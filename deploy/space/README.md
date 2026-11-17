---
title: AF Detect
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Explainable AF screening demo for single-lead ECG
---

# AF Detect

Upload a short single-lead ECG (9-61 s; a one-column `.csv` with its sampling rate, or a WFDB `.hea` + `.mat`/`.dat`
pair) or pick one of six examples from the held-out test data. The app answers **Normal rhythm**, **Atrial
fibrillation**, **Other rhythm** or **Too noisy to classify**, says **Uncertain** when it is not confident, and shows
why: the ECG with detected heartbeats, the time between beats and a Poincaré plot.

The model is a gradient-boosting classifier on 21 explainable features (heart-rhythm irregularity, P-wave strength,
beat shape, atrial activity and signal quality), trained on the PhysioNet/CinC Challenge 2017 data and frozen before
it was tested. It runs on CPU. Code, methods and all results (test split and external MIT-BIH AF Database):
https://github.com/SalemBenafia/af-ecg (see `results/RESULTS.md`).

**Research prototype from a student hackathon. Not a medical device. It cannot rule out heart disease. If you have
symptoms, contact a doctor or emergency services.**

Data: PhysioNet/CinC Challenge 2017 and MIT-BIH Atrial Fibrillation Database, open access (ODC Attribution licence)
[ADD SOURCE].
