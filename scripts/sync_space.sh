#!/usr/bin/env bash
# Build deploy/space/ from the repo and push it to the Hugging Face Space.
#
#   bash scripts/sync_space.sh --build-only   # rebuild + check deploy/space/, no network, no token
#   bash scripts/sync_space.sh                # rebuild + check, then clone/update ../af-ecg-space, commit, push
#
# The HF token is NEVER written into a URL, a file or an environment variable by this script.
# Only `git push` (the last step) needs it: git asks for Username (your HF user name) and Password
# (paste your HF write token). With no git credential helper configured, git does not store it.
set -euo pipefail

SPACE_URL="https://huggingface.co/spaces/Salem85/af-ecg"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="$REPO_ROOT/deploy/space"
CLONE="$(dirname "$REPO_ROOT")/af-ecg-space"
PY="$REPO_ROOT/.venv/bin/python"

fail() { echo "ERROR: $*" >&2; exit 1; }

# ---------- 1. build deploy/space (copies; the README/requirements/packages files are kept) ----------
cd "$REPO_ROOT"
rm -rf "$DEPLOY/app.py" "$DEPLOY/src" "$DEPLOY/models" "$DEPLOY/examples"
mkdir -p "$DEPLOY/src/afdetect" "$DEPLOY/models" "$DEPLOY/examples"
cp app.py "$DEPLOY/app.py"
rsync -a --exclude 'cnn.py' --exclude '__pycache__/' src/afdetect/ "$DEPLOY/src/afdetect/"
cp models/gbm.joblib models/abstain.json "$DEPLOY/models/"
cp examples/*.csv "$DEPLOY/examples/"
mkdir -p "$DEPLOY/results/explore"
cp results/explore/feature_medians_train.csv "$DEPLOY/results/explore/"   # reference values for the "Why" line
echo "Built $DEPLOY"

# ---------- 2. checks ----------
for f in README.md requirements.txt packages.txt app.py models/gbm.joblib models/abstain.json; do
  [[ -e "$DEPLOY/$f" ]] || fail "deploy/space/$f is missing"
done
[[ ! -e "$DEPLOY/src/afdetect/cnn.py" ]] || fail "cnn.py must not be deployed"
if grep -rnE '^\s*(import torch|from torch|from \.cnn|import afdetect\.cnn|from afdetect\.cnn)' "$DEPLOY" --include='*.py'; then
  fail "deploy/space imports torch or cnn"
fi
grep -q '^import spaces\|^    import spaces' "$DEPLOY/app.py" || fail "app.py must import spaces"
grep -q '@spaces.GPU' "$DEPLOY/app.py" || fail "app.py needs one @spaces.GPU function"
grep -qx 'spaces' "$DEPLOY/requirements.txt" || \
  fail "requirements.txt must list 'spaces' without a version (the ZeroGPU builder pins its own)"

# scikit-learn version: Space pin == version stored in the pickles == version in this environment.
"$PY" - "$DEPLOY" <<'EOF'
import re, sys
from pathlib import Path
import joblib, numpy, sklearn
deploy = Path(sys.argv[1])
pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", (deploy / "requirements.txt").read_text(), re.M))
stored = set(re.findall(rb"_sklearn_version.{1,12}?(\d+\.\d+\.\d+)", (deploy / "models/gbm.joblib").read_bytes()))
stored = {s.decode() for s in stored}
problems = []
if stored != {pins.get("scikit-learn")}:
    problems.append(f"gbm.joblib saved with scikit-learn {stored}, Space pins {pins.get('scikit-learn')}")
for name, mod in (("scikit-learn", sklearn), ("numpy", numpy), ("joblib", joblib)):
    if pins.get(name) != mod.__version__:
        problems.append(f"{name}: Space pins {pins.get(name)}, local environment has {mod.__version__}")
if pins.get("gradio") != "6.32.0":
    problems.append(f"gradio pin {pins.get('gradio')} != sdk_version 6.32.0")
if problems:
    sys.exit("ERROR: " + "; ".join(problems))
print(f"Versions OK: gbm.joblib saved with scikit-learn {stored.pop()}; Space pins scikit-learn "
      f"{pins['scikit-learn']}, numpy {pins['numpy']}, joblib {pins['joblib']}, gradio {pins['gradio']}")
EOF

# The deployed app must start from deploy/space alone (no `pip install -e .`) and never load torch.
(cd "$DEPLOY" && env -u PYTHONPATH AFDETECT_HOME="$DEPLOY" "$PY" -I -B -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('space_app', 'app.py')
app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
assert app.__file__.startswith('$DEPLOY'), app.__file__
assert sys.modules['afdetect'].__file__.startswith('$DEPLOY'), sys.modules['afdetect'].__file__
assert 'torch' not in sys.modules, 'torch was imported'
probs, text, _ = app.analyze(None, 300, 'AF_1')
print('Deployed app OK: afdetect from', sys.modules['afdetect'].__file__, '| AF_1 ->', text.splitlines()[0])
") || fail "deployed app does not start from deploy/space"
find "$DEPLOY" -name '__pycache__' -type d -prune -exec rm -rf {} +

if [[ "${1:-}" == "--build-only" ]]; then
  echo "Build-only: not touching $CLONE"
  exit 0
fi

# ---------- 3. clone or update the Space (a public Space needs no token for this) ----------
if [[ ! -d "$CLONE/.git" ]]; then
  git clone "$SPACE_URL" "$CLONE"
fi
cd "$CLONE"
git lfs install
git pull --ff-only

# ---------- 4. copy (keep the Space's .git/ and .gitattributes, delete everything else not in deploy/space) ----------
rsync -a --delete --exclude '.git/' --exclude '.gitattributes' --exclude '__pycache__/' "$DEPLOY/" "$CLONE/"
git add -A
if ! git lfs ls-files | grep -q 'models/gbm.joblib'; then
  fail "git lfs ls-files does not list models/gbm.joblib (check .gitattributes in the Space: *.joblib must use LFS)"
fi

if git diff --cached --quiet; then
  echo "No new changes to commit."
else
  SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
  git commit -m "Sync from af-ecg $SHA"
fi

# A commit from an earlier run that was interrupted (e.g. Ctrl+C at the password prompt) must still be pushed.
AHEAD="$(git rev-list --count '@{u}..HEAD')"
if [[ "$AHEAD" -eq 0 ]]; then
  echo "Space is already up to date (nothing to push)."
  exit 0
fi
echo "$AHEAD commit(s) to push."

# ---------- 5. push: the ONLY step that needs your HF token (typed at git's password prompt) ----------
echo "Pushing to $SPACE_URL: Username = your HF user name, Password = your HF write token (not stored)."
git push
