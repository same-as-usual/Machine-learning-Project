"""Leakage-safe train/val/test splits.

Splitting is done by `dup_group` (near-duplicate cluster), never by row:
a headline and its near-duplicate can never land in different splits.
Stratified by label so class balance is preserved.

When timestamps/outlets are available (Webis 2017, RSS scrape), this module
will additionally support time-based and outlet-grouped splits.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from manipulens.config import data_dir, load_params
from manipulens.data.schemas import validate_deduped, validate_split


def grouped_stratified_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> pd.Series:
    """Assign each dup_group to train/val/test, stratified by group majority label."""
    rng = np.random.default_rng(seed)
    group_label = df.groupby("dup_group")["label_clickbait"].agg(lambda s: int(s.mean() >= 0.5))
    assignment: dict[int, str] = {}
    for label in group_label.unique():
        groups = group_label[group_label == label].index.to_numpy().copy()
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, round(n * test_size)) if n > 2 else 0
        n_val = max(1, round(n * val_size)) if n > 2 else 0
        for g in groups[:n_test]:
            assignment[g] = "test"
        for g in groups[n_test : n_test + n_val]:
            assignment[g] = "val"
        for g in groups[n_test + n_val :]:
            assignment[g] = "train"
    return df["dup_group"].map(assignment)


def check_no_leakage(df: pd.DataFrame) -> None:
    """Every dup_group must live in exactly one split. Hard failure otherwise."""
    splits_per_group = df.groupby("dup_group")["split"].nunique()
    leaky = splits_per_group[splits_per_group > 1]
    if not leaky.empty:
        raise AssertionError(
            f"near-dup leakage across splits in groups: {leaky.index.tolist()[:10]}"
        )


def main(argv: list[str] | None = None) -> None:
    params = load_params()["splits"]
    df = pd.read_parquet(data_dir("interim") / "deduped.parquet")
    df = validate_deduped(df)

    df["split"] = grouped_stratified_split(
        df, test_size=params["test_size"], val_size=params["val_size"], seed=params["seed"]
    )
    check_no_leakage(df)
    df = validate_split(df)

    out_dir = data_dir("processed")
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        part.to_parquet(out_dir / f"{split}.parquet", index=False)
        balance = part["label_clickbait"].mean()
        print(f"{split}: {len(part)} rows (clickbait share {balance:.2%})")


if __name__ == "__main__":
    main(sys.argv[1:])
