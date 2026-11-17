# AF Detect: project memory for Claude Code

We build a screening aid that reads a short single-lead ECG (9-61 s) and classifies it as
N (normal), A (atrial fibrillation), O (other rhythm) or ~ (too noisy). It says "Uncertain" when it is not confident.
Built for the AI for Medicine hackathon. Deadline: Fri 4 Dec 2026, 11:45 PM ET. Public repo.
Judged on: Technical Depth, Rigor & Reproducibility, Clinical Relevance & Honesty, Demo Quality (25% each).
Full product and technical spec: docs/SPEC.md (read it at the start of every new phase).

## Stack
- Python 3.12, venv in .venv (uv), package `afdetect` in src/ (installed with `uv pip install -e .`)
- NeuroKit2 (cleaning, R peaks, quality), wfdb (PhysioNet files), numpy, scipy, pandas 3, scikit-learn, matplotlib
- Demo: Gradio 6 (app.py), hosted on a Hugging Face Space
- No deep learning: the team decided against the optional CNN (no PyTorch, no Colab)

## Commands
- `make test`: must pass before you say a task is done (= .venv/bin/python -m pytest -q)
- `make splits` / `make features` / `make train` / `make examples` / `make app`
- `make eval-test` and `make afdb`: ONLY after I say the models are frozen (git tag v1-frozen). Never on your own.
- `git push` and pushes to the Hugging Face Space: only when I ask.

## Hard rules (rigor is the product)
- IMPORTANT: NEVER tune, select or look at scores on the test split or AFDB before the freeze tag. Decide on VAL.
- Never regenerate or edit configs/splits.csv.
- MIT-BIH AFDB is external test data only: never train, tune or choose thresholds on it.
- Only report numbers that exist in results/ files. Never invent metrics, statistics or citations: write [ADD SOURCE].
- Never drop a record silently: a failed recording gets NaN features, not a missing row.
- The app is a screening/research demo, never a diagnosis: keep the disclaimer and the "Uncertain" (abstain) logic.
- Never write tokens (Hugging Face, GitHub) into files, prompts, notebooks or logs.

## Code rules
- All parameters in configs/*.yaml; all randomness seeded; outputs go to results/ and models/.
- Every new function gets a pytest test (synthetic ECGs from tests/synth.py are fine).
- pandas 3: string columns are not numpy dtypes; use pd.api.types helpers.
- Small, readable functions with docstrings: the team must explain every line to the judges.
- data/ is git-ignored. models/*.joblib are committed (the demo needs them).
- Everything runs on the laptop CPU. There is no CNN and no Colab step.

## Architecture (src/afdetect, SPEC section 5)
data.py (IO) -> features.py (300 Hz, nk.ecg_clean, nk.ecg_peaks, RR / P-wave / f-wave / beat-shape / quality)
-> extract.py (all records -> one CSV) -> models.py (rr_rule, logreg_rr, gbm) -> train.py (fit on train,
compare on val, abstain threshold) -> evaluate.py (test split or --external) -> afdb.py (30-s windows)
-> app.py (Gradio)

## Workflow
- For changes across several files, propose a short plan first and stay inside the current phase's scope.
- After changes, run `make test`, fix root causes (don't silence errors or skip tests), then summarise what
  changed and how I can check it by hand.
- When you make a plot, look at it yourself and tell me what you see.
- When compacting, keep the list of modified files and any open TODOs.
