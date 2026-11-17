#!/usr/bin/env bash
# Download both open-access PhysioNet databases into data/ (no login needed). Safe to re-run: skips what is there.
#   CinC 2017 training set -> data/training2017/   (zip checked against the SHA-256 of the version we used)
#   MIT-BIH AF Database    -> data/afdb/           (checked with PhysioNet's SHA256SUMS.txt)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p data

CINC_URL="https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"
CINC_SHA256="25fdbab36bb6724b66a4d2c26ff89f6b9d6ff87e6aed714d24cf5bf071cd1255"
AFDB_URL="https://physionet.org/files/afdb/1.0.0/"

if [[ -f data/training2017/REFERENCE.csv ]]; then
  echo "CinC 2017: data/training2017 already present, skipping download."
else
  [[ -f data/training2017.zip ]] || curl -fL --retry 3 -o data/training2017.zip "$CINC_URL"
  echo "$CINC_SHA256  data/training2017.zip" | sha256sum -c -
  unzip -q -o data/training2017.zip -d data/
fi
if [[ -f data/training2017.zip ]]; then
  echo "$CINC_SHA256  data/training2017.zip" | sha256sum -c -
fi
echo "CinC 2017 records: $(ls data/training2017/*.hea | wc -l) (expected 8528)"

if [[ "$(ls data/afdb/*.dat 2>/dev/null | wc -l)" -ge 23 ]]; then
  echo "AFDB: data/afdb already has the signal files, skipping download."
else
  wget -r -N -c -np -nH --cut-dirs=3 -P data/afdb "$AFDB_URL"
fi
(cd data/afdb && sha256sum --quiet -c SHA256SUMS.txt) && echo "AFDB: all files match SHA256SUMS.txt"
