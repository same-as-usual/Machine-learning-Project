"""LLM labeler tests — dry-run mode only (no API calls, no key needed)."""

import json

import pandas as pd

from manipulens.labeling.labeling_functions import DIMENSIONS
from manipulens.labeling.llm_labeler import (
    GOLD_FILE,
    LABEL_SCHEMA,
    build_system,
    label_headlines,
)


def test_schema_covers_all_dimensions():
    props = LABEL_SCHEMA["properties"]
    for dim in DIMENSIONS:
        assert props[dim]["enum"] == [0, 1, 2]
    assert LABEL_SCHEMA["additionalProperties"] is False
    assert set(LABEL_SCHEMA["required"]) == {*DIMENSIONS, "out_of_scope"}


def test_system_prompt_contains_codebook_and_cache_control():
    system = build_system()
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "Curiosity gap" in system[0]["text"]
    assert "0-2 ordinal" in system[0]["text"] or "0–2 ordinal" in system[0]["text"]


def test_dry_run_labels_shape(tmp_path, monkeypatch):
    import manipulens.labeling.llm_labeler as m

    monkeypatch.setattr(m, "CACHE_FILE", tmp_path / "cache.jsonl")
    df = label_headlines(
        ["You Won't Believe What Happened", "Fed holds rates steady"], dry_run=True
    )
    assert len(df) == 2
    for dim in DIMENSIONS:
        assert df[dim].isin([0, 1, 2]).all()
    # clickbait headline should out-score wire copy on curiosity
    assert df.iloc[0]["curiosity_gap"] > df.iloc[1]["curiosity_gap"]


def test_cache_prevents_relabeling(tmp_path, monkeypatch):
    import manipulens.labeling.llm_labeler as m

    monkeypatch.setattr(m, "CACHE_FILE", tmp_path / "cache.jsonl")
    label_headlines(["Some headline here"], dry_run=True)
    first = (tmp_path / "cache.jsonl").read_text()
    label_headlines(["Some headline here"], dry_run=True)
    assert (tmp_path / "cache.jsonl").read_text() == first  # no second append
    rec = json.loads(first)
    assert set(rec["labels"]).issuperset(DIMENSIONS)


def test_gold_seed_is_valid():
    gold = pd.read_csv(GOLD_FILE)
    assert len(gold) >= 25
    for dim in DIMENSIONS:
        assert gold[dim].isin([0, 1, 2]).all(), dim
    assert gold["headline"].is_unique
