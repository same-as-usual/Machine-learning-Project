"""Pandera data contracts. These run in CI (tests/data/) and inside the pipeline.

The schema is the single source of truth for what a "headline record" is at each
pipeline stage. Breaking it is a build failure, not a silent drift.
"""

from __future__ import annotations

import pandera as pa
from pandera import Check, Column

# Stage 1: raw ingested headlines
RawHeadlineSchema = pa.DataFrameSchema(
    {
        "headline": Column(str, Check.str_length(min_value=1), nullable=False),
        "label_clickbait": Column(int, Check.isin([0, 1]), nullable=False),
        "source": Column(str, nullable=False),  # dataset provenance, e.g. "chakraborty2016"
        "outlet": Column(str, nullable=True),  # publisher when known, else ""
    },
    strict=False,
    coerce=True,
)

# Stage 2: deduped (adds dedup bookkeeping)
DedupedSchema = RawHeadlineSchema.add_columns(
    {
        "dup_group": Column(int, nullable=False),  # near-duplicate cluster id
    }
)

# Stage 3: split (adds split assignment)
SplitSchema = DedupedSchema.add_columns(
    {
        "split": Column(str, Check.isin(["train", "val", "test"]), nullable=False),
    }
)


def validate_raw(df):  # type: ignore[no-untyped-def]
    return RawHeadlineSchema.validate(df)


def validate_deduped(df):  # type: ignore[no-untyped-def]
    return DedupedSchema.validate(df)


def validate_split(df):  # type: ignore[no-untyped-def]
    return SplitSchema.validate(df)
