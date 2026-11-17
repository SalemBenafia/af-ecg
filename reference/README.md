# Reference code (compare, don't copy blindly)

These are the reference files from Appendix D of the build manual. They are here so you can compare
them with what Claude Code writes for you, phase by phase. They are NOT part of your package: your own
code lives in src/afdetect/ and is written in phases P1-P11.

| File | Phase |
|---|---|
| src/afdetect/features.py | P3, P4 |
| src/afdetect/models.py, metrics.py, train.py | P6 |
| src/afdetect/cnn.py | P8 (optional) |
| src/afdetect/evaluate.py, afdb.py | P9 |
| app.py | P11 |

data.py, splits.py and extract.py are not included: their prompts (P1, P2, P5) describe them fully.

How they were checked: see the note at the start of Appendix D in the manual. In short, the original guide
ran them against 14 automated tests; this edition re-ran everything except cnn.py on synthetic ECGs with the
pinned packages (19 checks passed). Neither run used the real PhysioNet files: your real scores come from your
own run.
