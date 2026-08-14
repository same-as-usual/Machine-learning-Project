import numpy as np
import pandas as pd
import pytest

from manipulens.data.splits import check_no_leakage, grouped_stratified_split


def _fake_df(n_groups: int = 200, rows_per_group: int = 2) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        label = g % 2
        for _ in range(rows_per_group):
            rows.append({"headline": f"headline {g}", "label_clickbait": label, "dup_group": g})
    return pd.DataFrame(rows)


def test_groups_never_straddle_splits():
    df = _fake_df()
    df["split"] = grouped_stratified_split(df, test_size=0.15, val_size=0.15, seed=42)
    check_no_leakage(df)  # must not raise


def test_leakage_check_catches_straddling_group():
    df = _fake_df(n_groups=2)
    df["split"] = ["train", "test", "train", "train"]  # group 0 straddles
    with pytest.raises(AssertionError, match="leakage"):
        check_no_leakage(df)


def test_split_sizes_roughly_correct():
    df = _fake_df(n_groups=1000, rows_per_group=1)
    df["split"] = grouped_stratified_split(df, test_size=0.15, val_size=0.15, seed=42)
    shares = df["split"].value_counts(normalize=True)
    assert shares["test"] == pytest.approx(0.15, abs=0.03)
    assert shares["val"] == pytest.approx(0.15, abs=0.03)


def test_stratification_preserves_class_balance():
    df = _fake_df(n_groups=1000, rows_per_group=1)
    df["split"] = grouped_stratified_split(df, test_size=0.2, val_size=0.2, seed=0)
    overall = df["label_clickbait"].mean()
    for split in ("train", "val", "test"):
        share = df.loc[df["split"] == split, "label_clickbait"].mean()
        assert share == pytest.approx(overall, abs=0.05)


def test_deterministic_given_seed():
    df = _fake_df()
    a = grouped_stratified_split(df, 0.15, 0.15, seed=7)
    b = grouped_stratified_split(df, 0.15, 0.15, seed=7)
    assert np.array_equal(a.to_numpy(), b.to_numpy())
