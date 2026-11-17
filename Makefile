# One command per step. Run from the project root with the venv created by uv (.venv).
PY := .venv/bin/python
CFG := configs/train.yaml

.PHONY: test data splits features train eval-test afdb examples app

test:
	$(PY) -m pytest -q

# Download CinC 2017 and MIT-BIH AFDB from PhysioNet (open access, checksums verified).
data:
	bash scripts/download_data.sh

# The split is a file target with no prerequisites: once configs/splits.csv exists,
# `make splits` does nothing ("Nothing to be done"). Never delete or regenerate it.
splits: configs/splits.csv
configs/splits.csv:
	$(PY) -m afdetect.splits --root data/training2017 --out $@

features:
	$(PY) -m afdetect.extract --root data/training2017 --out data/features/cinc2017.csv --jobs -1

train:
	$(PY) -m afdetect.train --config $(CFG)

# Run ONCE, after `git tag v1-frozen` (phase P9).
eval-test:
	$(PY) -m afdetect.evaluate --config $(CFG) --split test

afdb:
	$(PY) -m afdetect.afdb --root data/afdb --out data/features/afdb.csv
	$(PY) -m afdetect.evaluate --config $(CFG) --external data/features/afdb.csv

examples:
	$(PY) scripts/make_examples.py

app:
	$(PY) app.py
