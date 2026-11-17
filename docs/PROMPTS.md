# AF Detect: every Claude Code prompt

Copy from this file rather than from the PDF. Type /clear before each new phase. "Plan mode first" means: start with /plan (panel) or Shift+Tab until plan mode is on (terminal), read the plan, correct it, then approve.

## P0a · Understand the project (plan mode, no code)  (plan mode first)

```text
Read CLAUDE.md and docs/SPEC.md. Do not write or change any code yet.
1. Explain back to me, in about 10 bullet points, what we are building, which data is used for what (train / val / test / AFDB), and which rules protect the test split.
2. List anything in the spec that is unclear, risky or contradictory, with your suggested decision for each.
3. Tell me which parts run on my laptop CPU, which part needs the Colab GPU, and why you cannot run the Colab part yourself.
```

## P0b · Project skeleton  (plan mode first)

```text
Set up the project skeleton following CLAUDE.md and docs/SPEC.md. Do not implement any algorithms yet.
- pyproject.toml: setuptools, package "afdetect" in src/, Python >=3.10, pytest pythonpath ["src", "tests"], filter DeprecationWarning/RuntimeWarning/UserWarning in pytest.
- requirements.txt pinned to: neurokit2==0.3.0, wfdb==4.3.1, scikit-learn==1.9.1, joblib==1.6.0, numpy==2.5.5, scipy==1.18.1, pandas==3.1.1, PyYAML==6.0.3, matplotlib==3.11.2, gradio==6.32.0, pytest==9.2.0. torch is NOT listed: it is only for the optional CNN and is preinstalled on Colab.
- .gitignore: .venv/, __pycache__/, *.egg-info/, data/, *.pt, results/**/predictions.csv (models/*.joblib stay in git: they are small and the demo needs them).
- Empty modules with one-line docstrings: src/afdetect/{__init__,data,features,splits,extract,models,metrics,train,evaluate,afdb,cnn}.py
- Keep the Makefile from the starter kit. Check that every target calls a module from this list.
- tests/test_smoke.py importing every module except cnn.
- LICENSE: the MIT licence, year 2026, copyright holder "[YOUR TEAM NAME]".
Install with: uv pip install -r requirements.txt && uv pip install -e .
Then run make test and show me the tree.
```

## P1 · Data loader and first look  (plan mode first)

```text
Implement src/afdetect/data.py (SPEC section 5): LABELS = ["N","A","O","~"], NAMES (human-readable), load_reference(root) reading REFERENCE.csv (no header; assert only the 4 labels appear), read_wfdb(path_without_ext, channel=0) returning (signal in mV, fs) with wfdb.rdrecord, read_csv_ecg(path, fs) for a one-column CSV with an optional header (pandas 3: use pd.api.types.is_numeric_dtype), and read_uploaded(paths, csv_fs) for the app (.csv, or .hea + .mat/.dat).
Add tests using wfdb.wrsamp to write fake records into tmp_path.
Then write scripts/explore.py: print the class counts and the recording-length distribution of data/training2017, and save results/explore/examples.png with 10 seconds of 2 random recordings per class (4 rows). Look at the PNG with me and describe what differs between the classes.
```

## P2 · Stratified split (once, forever)

```text
Implement src/afdetect/splits.py with make_splits(ref, seed=42, val=0.15, test=0.15) using sklearn train_test_split twice with stratify on the label, and a CLI: python -m afdetect.splits --root data/training2017 --out configs/splits.csv (columns record,label,split). The CLI must refuse to overwrite an existing file. Print a split x label table. Add a test on a fake reference table checking the sizes and that each split has every class. Then run make splits on the real data. Never regenerate the split after today.
```

## P3 · Preprocessing, synthetic ECGs and beat detection  (plan mode first)

```text
In src/afdetect/features.py implement (SPEC section 3, step 2): FS = 300; to_fs(ecg, fs) using resample_poly with a Fraction (250 -> 300 is 6/5); preprocess(ecg, fs) -> (raw at 300 Hz minus median, cleaned with nk.ecg_clean method "neurokit", R-peak indices from nk.ecg_peaks; on any exception return an empty int array).
Create tests/synth.py with make_ecg(kind, seconds, fs, seed) that builds synthetic ECGs from Gaussian P, Q, R, S, T waves: "N" regular RR with mild variation; "A" RR = mean x uniform(0.6, 1.4), no P wave, plus 5-7 Hz f-waves; "O" regular rhythm with a premature wide beat every 2nd or 3rd beat (short RR then compensatory long RR, no P wave on the premature beat); "~" strong noise, baseline wander and motion bursts.
Tests: heart rate of a synthetic normal ECG is within 50-100 bpm; 250 Hz resampling gives the right length; a flat signal does not crash.
Then write scripts/check_peaks.py that plots 10 s of 8 random REAL recordings (2 per class) with the detected R peaks marked, to results/explore/peaks.png. Look at it and tell me where peak detection fails.
```

## P4 · Feature extraction + tests  (plan mode first)

```text
Implement in src/afdetect/features.py: rr_features(rpeaks), morphology_features(cleaned, rpeaks), quality_features(raw, cleaned, rpeaks), extract_features(ecg, fs) and the lists FEATURES and RR_FEATURES, exactly as defined in SPEC section 6. Details:
- RR: keep intervals between 0.25 and 2.5 s; if fewer than 5, return NaN for the RR features; sample entropy with nk.entropy_sample(rr, delay=1, dimension=2, tolerance=0.2*std) and cap infinite values at 3.0; Shannon entropy of a 16-bin histogram normalised by log2(16); turning-point ratio; SD1/SD2 of the Poincare plot.
- Morphology: beats from 0.25 s before to 0.45 s after each R peak; median template; p_wave_ratio = peak-to-peak of the detrended template from -0.25 s to -0.08 s divided by QRS peak-to-peak (+-0.05 s); tmpl_corr = mean correlation of beats with the template; fwave_ratio = Welch power 4-9 Hz over 1-40 Hz of the signal after subtracting the template at every beat.
- Quality: Welch power >40 Hz and <0.7 Hz ratios, kurtosis, nk.ecg_quality mean (NaN if <4 peaks), 1-99 percentile range, duration.
Tests on synthetic ECGs (8 per class, median per class): AF rmssd_norm > 3x Normal; AF p_wave_ratio < 0.5x Normal; AF fwave_ratio > Normal; Other rr_entropy < AF; Noisy tmpl_corr < Normal - 0.3 and hf_noise_ratio > 10x Normal.
Finally, explain each feature in one simple sentence and save the sentences in docs/FEATURES.md (I will use them in my write-up).
```

## P5a · Feature extraction CLI (test on 50 records)

```text
Write src/afdetect/extract.py: python -m afdetect.extract --root data/training2017 --out data/features/cinc2017.csv [--jobs -1] [--limit N]. Use joblib Parallel over the records in REFERENCE.csv, suppress warnings inside the worker, and merge with the labels. Print the class counts and the number of NaN values per feature at the end. A recording that fails must become a row of NaN features, never a missing row.
Add a test on fake WFDB records in tmp_path. Run the CLI with --limit 50 only: I will run the full extraction myself with make features.
Then write scripts/feature_plots.py: one box plot per feature by class (results/explore/features_by_class.png) and a table of the median per class, using the TRAIN split only. Don't run it yet.
```

## P5b · Sanity plots on the full features CSV

```text
The full features CSV is ready in data/features/cinc2017.csv. Run scripts/feature_plots.py, look at the plot, and tell me which 5 features separate AF from Normal best, and which separate AF from Other (TRAIN split only).
Flag any feature whose box plot looks physiologically wrong.
```

## P6 · Models, metrics and training CLI  (plan mode first)

```text
Implement (SPEC sections 7, 8 and 9):
- src/afdetect/metrics.py: summarize(y_true, y_pred, p_af) -> F1 per class (labels N, A, O, ~), challenge_F1 = mean F1 of N, A, O, accuracy, AF sensitivity/specificity/PPV, AF AUROC when possible; confusion(); bootstrap_ci(y_true, y_pred, metric, groups=None, n_boot, seed) resampling recordings or groups; coverage_curve(y_true, y_pred, confidence, thresholds).
- src/afdetect/models.py: class RRRule (fit chooses the rmssd_norm threshold maximising AF F1 on train; predict_proba returns Normal/AF columns), make_model(name, seed) for "rr_rule", "logreg_rr", "gbm" (HistGradientBoostingClassifier max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, class_weight="balanced", early_stopping=True, random_state=seed), and predict(model, cols, X) returning labels + probabilities in LABELS order.
- src/afdetect/train.py + configs/train.yaml (from the starter kit): train each model on TRAIN, evaluate on VAL with CIs, save models/<name>.joblib, write results/val/summary.json, and for gbm choose the abstain threshold = the highest threshold that still keeps >= 90% of VAL recordings; save models/abstain.json with the curve.
Tests: hand-computed metric values on a tiny example; bootstrap CI contains the point estimate.
Then run make train and show me a VAL table. Do not touch the test split.
```

## P7 · Ablation study on VAL

```text
Using ONLY train (fit) and val (score), run an ablation for the gbm model with these feature sets: RR only / RR + P wave / RR + P wave + f-wave / all features. Also try class_weight None vs "balanced", and learning_rate 0.05 vs 0.1. Use 3 different seeds for each and report mean +- std of the val challenge F1 and the AF sensitivity. Add sklearn permutation_importance (on val) for the best model. Save results/val/ablation.md with a table and a short recommendation (3 sentences). Do not change configs/train.yaml yourself: I decide. Do not run anything on the test split.
```

## P8 · CNN module and Colab notebook  (plan mode first)

```text
Implement src/afdetect/cnn.py (PyTorch) as described in SPEC section 7: WIN = 9000; load_signals (wfdb + resample to 300 Hz + per-record z-score); ECGSet dataset (random crop + random sign flip using torch RNG when training, centre crop when evaluating, zero-pad short records); ResNet1D (stem conv k15 s2 + BN + ReLU + maxpool, then for widths 32, 64, 128, 256: Block(stride 2) + Block(stride 1), adaptive average pool, dropout 0.3, linear to 4); training with AdamW, CosineAnnealingLR, class-weighted CrossEntropyLoss, torch.autocast + torch.amp.GradScaler only on CUDA, saving the best-val-F1 weights; a --test flag that loads the best weights and evaluates the test split ONCE. Config: configs/cnn.yaml (from the starter kit).
Add tests/test_cnn.py that is skipped if torch is missing (pytest.importorskip): output shape (2, 4) for a (2, 1, 9000) input, and a 2-epoch CPU run on 40 synthetic records.
Then create notebooks/colab_cnn.ipynb with short commented cells: GPU check; mount Drive; clone/pull my public repo; pip install wfdb neurokit2 and pip install -e . --no-deps; download training2017.zip from PhysioNet to /content/data and unzip; run the feature tests; train; plot the VAL F1 per epoch; and a final cell, clearly marked "run only once at the end", for --test.
```

## P9a · Evaluation code for test + AFDB (fake data only)  (plan mode first)

```text
Write the final evaluation code (SPEC sections 4 and 8). Test it on fake data only: do NOT run it on the real test split or on data/afdb.
1. src/afdetect/evaluate.py: python -m afdetect.evaluate --config configs/train.yaml --split test, or --external <features.csv>. For every model in the config: summarize() + 95% bootstrap CIs for challenge F1 and AF sensitivity, and the confusion matrix. For gbm also: coverage at the abstain threshold from models/abstain.json, confusion_gbm.png and predictions.csv (record, label, pred, p_af, confidence). Write results/<name>/summary.json and RESULTS_table.md. With --external, bootstrap by source_record (windows from the same person are not independent).
2. src/afdetect/afdb.py: python -m afdetect.afdb --root data/afdb --out data/features/afdb.csv. For each record that has a .dat file: read channel 0 with wfdb (250 Hz), read the .atr rhythm annotations (aux_note "(AFIB" -> A, "(N" -> N, "(AFL" and "(J" -> O), build rhythm intervals, cut non-overlapping 30-s windows that lie completely inside one rhythm, randomly keep at most 60 windows per label per record (fixed seed), skip windows with NaN or flat signal, and extract features (they resample to 300 Hz). Store record = <record>_<start> and source_record.
3. Tests: write a fake 250 Hz 2-lead record with wfdb.wrsamp and rhythm annotations with wfdb.wrann (symbol "+", aux_note "(N", "(AFIB", "(AFL") and check the window counts; run evaluate.py on a tiny synthetic split in tmp_path and check that every output file exists.
Report AF sensitivity and specificity as the main external numbers, because AFDB's "(N" means "all other rhythms", not "normal". Run make test.
```

## P9b · Write the results page

```text
Read results/test/summary.json, results/afdb/summary.json and (if present) the CNN test summary. Write results/RESULTS.md: (1) test table: all models with challenge F1 (95% CI), F1 per class, AF sensitivity (95% CI), specificity, PPV, AUROC; (2) the gbm confusion matrix image; (3) the abstain result: % of recordings kept and the score on them; (4) external AFDB: AF sensitivity/specificity with CIs, and how much they changed compared with the CinC test; (5) 5 honest sentences about what the numbers mean.
Use only numbers that exist in the files. Do not change any model or setting after this.
```

## P10 · Subgroups and failure cases

```text
Using results/test/predictions.csv and the features CSV (no retraining): report the challenge F1 and AF sensitivity by recording length (<20 s, 20-40 s, >40 s), by heart rate (<60, 60-100, >100 bpm, from hr_mean), and by signal quality (nk_quality terciles), with the number of recordings per group. Then plot 6 misclassified test recordings (AF missed, false AF alarms, AF called Other) with R peaks marked, to results/errors/*.png, and write results/ERRORS.md with one sentence per plot about the likely cause (look at the plots). Small groups: say clearly that they are descriptive only.
```

## P11 · Examples and Gradio app  (plan mode first)

```text
Build the demo described in SPEC section 10.
1) scripts/make_examples.py: pick (seed 7) 2 Normal, 2 AF, 1 Other and 1 Noisy recording from the TEST split, save each as a one-column CSV at 300 Hz in examples/ (Normal_1.csv, AF_1.csv, ...) and write examples/README.md mapping each file to its record and true label.
2) app.py (Gradio 6): add src/ to sys.path; load models/gbm.joblib and models/abstain.json; inputs: gr.File(file_count="multiple", types .csv/.hea/.mat/.dat), gr.Number for the CSV sampling rate (default 300), gr.Dropdown of examples (default "(none)"), Analyze button; analyze(): read the input, refuse recordings shorter than 9 s, extract features, predict, show gr.Label with the 4 probabilities, a markdown verdict (or "Uncertain" when the top probability is below the threshold) with average heart rate and a "Why" line, and one matplotlib figure with 3 panels (10 s of ECG with R peaks, RR intervals in ms by beat number, Poincare plot). Visible disclaimer (SPEC section 11).
3) tests/test_app.py: train a tiny gbm on synthetic features in tmp_path, chdir there, load app.py with importlib, and check: the AF example is recognised, an uploaded .hea/.dat pair of a synthetic normal ECG gives "Normal rhythm", a 5-s CSV raises "too short". Run the tests, then python app.py.
```

## P11b · Fix from a screenshot (repeat as needed)

```text
[paste or Shift+drag a screenshot here]
This is the demo app at [window width] with the [example name] example loaded. Problems I see: [describe them]. Fix them without changing the model, the abstain threshold or the disclaimer. Tell me what you changed.
```

## P12 · Deploy to Hugging Face

```text
Prepare deploy/space/ with exactly what the Space needs: app.py, src/afdetect/ (without cnn.py), models/gbm.joblib, models/abstain.json, examples/*.csv, requirements.txt (repo pins without pytest, plus "spaces"), packages.txt and README.md with the front matter from the manual and a short description + disclaimer. Write scripts/sync_space.sh that copies these files into a local clone of https://huggingface.co/spaces/YOUR_USER/af-ecg and pushes. Tell me which commands need my HF token, and never write the token into a file. Check that the scikit-learn version in the Space matches the one that saved the .joblib files.
```

## P13a · README for the judges  (plan mode first)

```text
Write README.md for judges with 3 minutes: (1) summary + demo link + video link (placeholders); (2) the main results table from results/RESULTS.md (test split and AFDB) + confusion matrix; (3) install with uv / Python 3.12 and "reproduce in 4 commands" (make test, download data, make features train, make eval-test afdb); (4) data: both databases with their required citations and the PhysioNet citation, licence ODC-By; (5) method in 6 bullets with the pipeline figure; (6) limitations from SPEC section 11: label noise and expert disagreement, no patient IDs in CinC 2017, AF vs Other confusion, drop on AFDB, single lead only, screening not diagnosis; (7) attribution for NeuroKit2, wfdb, scikit-learn and, if used, PyTorch; (8) one honest AI-tool disclosure line; (9) MIT licence. Only numbers that exist in the results files.
```

## P13b · Mock judging (a fresh pair of eyes)

```text
Use a subagent so the review starts with fresh eyes. Act as a strict judge of the AI for Medicine hackathon.
Read the whole repo, README, docs/WRITEUP.md (if it exists) and results/. Score 0-10 on Technical Depth, Rigor & Reproducibility, Clinical Relevance & Honesty and Demo Quality, quoting the evidence you used. List the 5 changes that would raise the total most, in order, with the effort for each in minutes. Check every number in the README against the results files and flag mismatches. Check that results/test was created after the v1-frozen tag. Do not edit files.
```

## P13c · Fresh-clone reproducibility check

```text
Clone the repo from GitHub into /tmp/fresh_check, create a new Python 3.12 venv with uv, install, download the CinC 2017 data and AFDB, run make test, make features, make train, make eval-test and make afdb, and compare the numbers with results/RESULTS.md. Report differences and any missing or wrong step in the README, then fix the README. Do not change any code, model or setting.
```

## P14 · Devpost write-up draft

```text
Draft docs/WRITEUP.md with exactly these sections: The problem and who has it / Method / Evaluation (metric and how it is measured) / Who benefits, and who could be harmed / Limitations / What we would do with another month. Mark every place that needs a sourced statistic (for example how common AF is, stroke risk) with [ADD SOURCE]. In "who could be harmed" cover: missed AF (false reassurance), false alarms (anxiety, unnecessary visits), people whose rhythms are under-represented in the data, misuse as a diagnosis, and say what the app does about each (abstain, disclaimer, examples of failures). Max 900 words, simple English.
Only numbers that exist in the results files.
```

## Commit (after every phase)

```text
Show me a short summary of git diff --stat, then commit all changes with a clear conventional message for this phase (for example "feat: RR, P-wave and f-wave features with tests") and push to origin.
Never commit data/ or any token.
```

## When something breaks

```text
[paste the exact error message, the full traceback or a screenshot]
This happens when I [describe the steps; say if it came from the terminal or a Colab cell]. Find the root cause, fix it, and prove the fix by running the relevant test. Don't suppress the error or skip the test.
```
