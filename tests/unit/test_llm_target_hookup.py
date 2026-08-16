"""The validation gate must be ENFORCED at the training boundary:
LLM labels only override weak labels for dimensions that cleared alpha."""

import json

import pandas as pd

from manipulens.labeling.labeling_functions import DIMENSIONS
from manipulens.models.transformer import load_llm_taxonomy, taxonomy_targets

HEADLINE = "You Won't Believe What This Senator Said Next"


def _write_fixtures(tmp_path, usable):
    labels = tmp_path / "taxonomy_labels.parquet"
    pd.DataFrame(
        [{"headline": HEADLINE, **dict.fromkeys(DIMENSIONS, 2), "out_of_scope": False}]
    ).to_parquet(labels, index=False)
    report = tmp_path / "llm_label_validation.json"
    report.write_text(json.dumps({"usable_dimensions": usable}))
    return labels, report


def test_missing_files_mean_no_llm_labels(tmp_path):
    mapping, usable = load_llm_taxonomy(tmp_path / "nope.parquet", tmp_path / "nope.json")
    assert mapping == {} and usable == []


def test_usable_dimension_overrides_weak_label(tmp_path):
    labels, report = _write_fixtures(tmp_path, usable=["fear"])
    mapping, usable = load_llm_taxonomy(labels, report)
    targets = taxonomy_targets(HEADLINE, mapping, usable)
    fear_i = DIMENSIONS.index("fear")
    # LF says 0 fear for this headline; validated LLM label (2 -> 1.0) wins
    assert targets[fear_i] == 1.0


def test_unusable_dimension_keeps_weak_label(tmp_path):
    labels, report = _write_fixtures(tmp_path, usable=["fear"])
    mapping, usable = load_llm_taxonomy(labels, report)
    targets = taxonomy_targets(HEADLINE, mapping, usable)
    cur_i = DIMENSIONS.index("curiosity_gap")
    # curiosity_gap did NOT clear the gate -> weak label preserved (0.6, not 1.0)
    assert targets[cur_i] == 0.6


def test_empty_usable_list_disables_all_llm_labels(tmp_path):
    labels, report = _write_fixtures(tmp_path, usable=[])
    mapping, usable = load_llm_taxonomy(labels, report)
    assert mapping == {}
    assert taxonomy_targets(HEADLINE, mapping, usable) == taxonomy_targets(HEADLINE)


def test_out_of_scope_rows_excluded(tmp_path):
    labels = tmp_path / "labels.parquet"
    pd.DataFrame(
        [{"headline": HEADLINE, **dict.fromkeys(DIMENSIONS, 2), "out_of_scope": True}]
    ).to_parquet(labels, index=False)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"usable_dimensions": list(DIMENSIONS)}))
    mapping, _ = load_llm_taxonomy(labels, report)
    assert HEADLINE not in mapping


def test_unlabeled_headline_falls_back_to_weak_labels(tmp_path):
    labels, report = _write_fixtures(tmp_path, usable=list(DIMENSIONS))
    mapping, usable = load_llm_taxonomy(labels, report)
    other = "Federal Reserve holds interest rates steady"
    assert taxonomy_targets(other, mapping, usable) == taxonomy_targets(other)
