# AF Detect: product and technical spec (v1)

## 1. Pitch
A free, explainable screening aid: upload a short single-lead ECG and see whether the rhythm looks
Normal, like atrial fibrillation (AF), like another rhythm, or too noisy to judge, with plots that show why.
When it is not confident, it says "Uncertain" instead of guessing.

## 2. Users and intended use
- Users: people exploring low-cost AF screening with handheld single-lead ECG devices or some smartwatches;
  students and clinicians reviewing what such a tool can and cannot do.
- Intended use: research and screening demo. A positive or uncertain answer means "a clinician should review
  this recording". It is never a diagnosis and never for emergencies.

## 3. Pipeline
1. **Read**: one lead, any sampling rate, resampled to 300 Hz.
2. **Clean + beats**: `nk.ecg_clean(method="neurokit")`, then `nk.ecg_peaks` for R peaks. A flat or very short
   signal gives empty peaks and NaN features: never a crash, never a dropped record.
3. **Features**: 21 explainable numbers (section 6).
4. **Model**: probabilities for N, A, O, ~ (section 7).
5. **Abstain**: top probability below the threshold gives "Uncertain" (section 9).
6. **Evaluate**: VAL for every decision; the test split and AFDB once, after the freeze (section 8).
7. **Demo**: Gradio app (section 10).

## 4. Data
| Data set | Content | Role |
|---|---|---|
| PhysioNet/CinC Challenge 2017 training set | 8,528 AliveCor recordings, single lead, 300 Hz, 9-61 s. N 5,050 / A 738 / O 2,456 / ~ 284. Labels (version 3) in REFERENCE.csv (REFERENCE-original.csv, with the first-release labels, is not used) | train / val / test |
| MIT-BIH Atrial Fibrillation Database (AFDB) | 25 long-term recordings (10 h each), 2 ECG leads at 250 Hz, 23 with signal files; rhythm annotations in .atr | external test only |

- Labels: N normal, A AF, O other rhythm, ~ too noisy.
- Split: configs/splits.csv (columns record,label,split), stratified by label, 70/15/15 train/val/test,
  seed 42. Created once in P2, never regenerated. The public files have no patient IDs, so the same person
  may appear in train and test (a limitation; one more reason for the external AFDB test).
- AFDB labels from the .atr aux_note: "(AFIB" -> A, "(N" -> N, "(AFL" and "(J" -> O. AFDB "(N" means
  "all other rhythms", not "normal", so the main external numbers are AF sensitivity and specificity.
- AFDB windows: non-overlapping 30-s windows that lie completely inside one rhythm; at most 60 per label per
  record (fixed seed); windows with NaN or a flat signal are skipped; first ECG channel only.
- Both databases are open access (ODC Attribution licence). Cite both databases and the PhysioNet
  standard citation in the README.

## 5. Modules (src/afdetect)
| File | Exports |
|---|---|
| `data.py` | `LABELS`, `NAMES`, `load_reference(root)`, `read_wfdb(path, channel=0)`, `read_csv_ecg(path, fs)`, `read_uploaded(paths, csv_fs)` |
| `splits.py` | `make_splits(ref, seed=42, val=0.15, test=0.15)`; CLI writes configs/splits.csv and refuses to overwrite it |
| `features.py` | `FS`, `to_fs`, `preprocess`, `rr_features`, `morphology_features`, `quality_features`, `extract_features`, `FEATURES`, `RR_FEATURES` |
| `extract.py` | CLI: every record in REFERENCE.csv -> data/features/cinc2017.csv (joblib Parallel) |
| `models.py` | `RRRule`, `make_model(name, seed)`, `predict(model, cols, X)` |
| `metrics.py` | `summarize`, `confusion`, `bootstrap_ci`, `coverage_curve` |
| `train.py` | fit on TRAIN, compare on VAL; writes models/*.joblib, models/abstain.json, results/val/summary.json |
| `evaluate.py` | `--split test` or `--external <features.csv>`; writes results/<name>/summary.json, RESULTS_table.md, confusion_gbm.png, predictions.csv |
| `afdb.py` | AFDB -> 30-s windows -> data/features/afdb.csv (record, source_record, label, features) |
| `app.py` (repo root) | Gradio demo |

## 6. Features (21)
RR group: keep RR intervals strictly between 0.25 s and 2.5 s. If fewer than 5 remain, every RR feature
except n_beats and frac_rr_implausible is NaN.

| Group | Feature | Definition | AF tends to be |
|---|---|---|---|
| RR | n_beats | number of detected R peaks | |
| RR | frac_rr_implausible | fraction of RR intervals outside (0.25, 2.5) s | |
| RR | hr_mean | 60 / mean RR, in bpm | |
| RR | rr_cv | SD / mean of RR | high |
| RR | rmssd_norm | root mean square of successive RR differences / mean RR | high |
| RR | pnn50 | fraction of successive RR differences > 50 ms | high |
| RR | rr_mad_norm | median absolute deviation / median RR | high |
| RR | sd1_sd2 | Poincare SD1 / SD2 (round cloud = close to 1) | high |
| RR | tpr | turning-point ratio (random series = about 2/3) | near 2/3 |
| RR | rr_entropy | Shannon entropy of a 16-bin RR histogram / log2(16) | high (Other: lower) |
| RR | sampen | sample entropy: `nk.entropy_sample(rr, delay=1, dimension=2, tolerance=0.2*SD)`; infinite -> 3.0 | high |
| P wave | p_wave_ratio | peak-to-peak of the detrended median beat from -0.25 to -0.08 s / QRS peak-to-peak (+-0.05 s) | low |
| Beat shape | tmpl_corr | mean correlation of each beat with the median beat | |
| Beat shape | qrs_amp | QRS peak-to-peak of the median beat | |
| Atrial activity | fwave_ratio | Welch power 4-9 Hz / 1-40 Hz after subtracting the median beat at every beat | high |
| Quality | duration_s | recording length in seconds | |
| Quality | kurtosis | kurtosis of the cleaned signal | |
| Quality | nk_quality | mean of `nk.ecg_quality` (NaN if fewer than 4 peaks) | |
| Quality | hf_noise_ratio | Welch power above 40 Hz / total (raw signal) | |
| Quality | lf_ratio | Welch power below 0.7 Hz / total (raw signal) | |
| Quality | amp_range | 1-99 percentile range of the cleaned signal | |

Beats are cut from 0.25 s before to 0.45 s after each R peak; the median beat ("template") needs at least
3 complete beats, otherwise the beat-shape, P-wave and f-wave features are NaN.

## 7. Models
| Name | Definition | Inputs |
|---|---|---|
| rr_rule (baseline 1) | AF if rmssd_norm > threshold, else Normal. Threshold = the value (quantiles 0.50-0.99 of TRAIN) that maximises AF F1 on TRAIN | rmssd_norm |
| logreg_rr (baseline 2) | median imputer + standard scaler + LogisticRegression(max_iter=3000, class_weight="balanced") | the 11 RR features |
| gbm (main) | HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, class_weight="balanced", early_stopping=True, validation_fraction=0.15, random_state=seed) | all 21 features |

## 8. Evaluation protocol
- Main score: challenge F1 = mean of the F1 scores of N, A and O (the official 2017 score; ~ is excluded).
- Also: F1 per class, accuracy, AF sensitivity, AF specificity, AF PPV, AF AUROC (p(AF) vs rest), confusion matrix.
- 95% CIs: bootstrap with 2,000 resamples of recordings; for AFDB, resample source records (windows from the
  same person are not independent).
- Every decision (features, settings, main model, abstain threshold) is made on VAL.
- Freeze: commit, `git tag v1-frozen`, push. Then run once: `make eval-test` and `make afdb`.
- After the freeze you may describe errors (P10). You may not change models or settings because of them.

## 9. Abstain rule
- confidence = the highest of the gbm's four class probabilities.
- threshold = the highest value in 0.30, 0.35, ..., 0.95 that still keeps at least 90% of VAL recordings
  (`abstain_min_coverage_pct` in configs/train.yaml). Saved with the coverage curve in models/abstain.json.
- Report for test and AFDB: % of recordings kept, and the score on the kept recordings.

## 10. Demo app (app.py)
- Inputs: upload a .csv (one column) with a sampling-rate box (default 300 Hz), or a .hea + .mat/.dat pair;
  or pick an example (2 Normal, 2 AF, 1 Other, 1 Noisy, all from the TEST split, chosen with seed 7).
- Refuses recordings shorter than 9 s.
- Shows: probabilities for the 4 classes; the verdict, or "Uncertain" below the threshold; average heart rate;
  a "Why" line (beat-to-beat variability, P-wave strength, beat-shape consistency); one figure with 3 panels
  (10 s of ECG with detected beats, RR intervals by beat number, Poincare plot).
- The disclaimer is always visible.

## 11. Honesty promises (app, README and write-up)
- "Research prototype from a student hackathon. Not a medical device. It cannot rule out heart disease.
  If you have symptoms, contact a doctor or emergency services."
- Limitations we always report: label noise and expert disagreement (labels version 3); no patient IDs in
  CinC 2017; AF vs Other confusion; the change on AFDB; single lead only; the Noisy class is small
  (284 recordings, 3.3%, wide CIs); screening, not diagnosis.
- Every number comes from a results file. Every outside statistic has a source, or [ADD SOURCE].

## 12. Acceptance criteria (end to end)
- `make test` passes on a fresh clone with Python 3.12 and the pinned requirements.
- configs/splits.csv has 8,528 rows, every split has all 4 classes, and it is unchanged since P2.
- results/test/ and results/afdb/ exist and were created after the v1-frozen tag.
- results/RESULTS.md: every model with challenge F1 (95% CI), F1 per class, AF sensitivity (95% CI),
  specificity, PPV and AUROC; the gbm confusion matrix; abstain coverage; AFDB external result with CIs.
- Public demo: in a private browser window, each example returns an answer within seconds, the disclaimer
  is visible, and a 5-s recording is refused.
- Every number in the README and the write-up matches a results file exactly.
