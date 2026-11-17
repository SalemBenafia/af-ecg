"""Tests for afdetect.splits on a fake reference table (never the real configs/splits.csv)."""
import pandas as pd
import pytest

from afdetect.data import LABELS
from afdetect.splits import SPLITS, main, make_splits, split_table

# Roughly the real class balance, scaled down to 1,000 records.
COUNTS = {"N": 590, "A": 90, "O": 285, "~": 35}


def fake_reference():
    """A fake REFERENCE table with 1,000 records and an imbalanced label mix."""
    labels = [lab for lab, n in COUNTS.items() for _ in range(n)]
    return pd.DataFrame({"record": [f"A{i:05d}" for i in range(len(labels))], "label": labels})


def test_sizes_columns_and_disjoint():
    ref = fake_reference()
    s = make_splits(ref)
    assert list(s.columns) == ["record", "label", "split"]
    assert s["record"].tolist() == ref["record"].tolist()  # same records, same order, none dropped
    sizes = s["split"].value_counts()
    assert sizes.sum() == 1000
    assert sizes["test"] == 150
    assert sizes["val"] == 150
    assert sizes["train"] == 700


def test_every_split_has_every_class_in_proportion():
    s = make_splits(fake_reference())
    table = pd.crosstab(s["split"], s["label"])
    for split in SPLITS:
        for label in LABELS:
            assert table.loc[split, label] > 0
    # stratified: each split's AF share is within 1 percentage point of the overall share
    share = s.groupby("split")["label"].apply(lambda x: (x == "A").mean())
    assert (share - COUNTS["A"] / 1000).abs().max() < 0.01


def test_same_seed_same_split_and_different_seed_differs():
    ref = fake_reference()
    assert make_splits(ref, seed=42).equals(make_splits(ref, seed=42))
    assert not make_splits(ref, seed=42).equals(make_splits(ref, seed=1))


def test_split_table_has_totals():
    table = split_table(make_splits(fake_reference()))
    assert list(table.index) == SPLITS + ["total"]
    assert list(table.columns) == LABELS + ["total"]
    assert table.loc["total", "total"] == 1000


def test_cli_writes_once_and_refuses_overwrite(tmp_path, capsys):
    ref = fake_reference()
    ref.to_csv(tmp_path / "REFERENCE.csv", header=False, index=False)
    out = tmp_path / "splits.csv"
    main(["--root", str(tmp_path), "--out", str(out)])
    first = out.read_text()
    assert "total" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        main(["--root", str(tmp_path), "--out", str(out)])
    assert out.read_text() == first
