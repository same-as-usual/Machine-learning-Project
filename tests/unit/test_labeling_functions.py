from manipulens.labeling.labeling_functions import (
    DIMENSIONS,
    FEATURE_NAMES,
    feature_vector,
    score_headline,
)


def test_all_dimensions_scored():
    results = score_headline("Senate passes infrastructure bill in 69-30 vote")
    assert set(results.keys()) == set(DIMENSIONS)


def test_neutral_headline_scores_low():
    results = score_headline("Federal Reserve holds interest rates steady")
    assert all(res.score == 0.0 for res in results.values())


def test_curiosity_gap_detected_with_spans():
    res = score_headline("You Won't Believe What This Senator Said Next")["curiosity_gap"]
    assert res.score >= 0.6
    assert any("believe" in s.text.lower() for s in res.spans)


def test_outrage_detected():
    res = score_headline("Governor DESTROYS whiny critics in epic takedown")["outrage"]
    assert res.score >= 0.6


def test_fear_detected():
    res = score_headline("The silent killer hiding in your kitchen")["fear"]
    assert res.score >= 0.6


def test_false_certainty_detected():
    res = score_headline("Scientists prove this food ruins your sleep")["false_certainty"]
    assert res.score >= 0.6


def test_sensational_formatting_caps_and_punct():
    res = score_headline("27 INSANE budget facts!! #9 will blow your mind")[
        "sensational_formatting"
    ]
    assert res.score >= 0.6
    assert len(res.spans) >= 2


def test_acronyms_not_flagged_as_caps():
    res = score_headline("FBI and CDC announce joint investigation")["sensational_formatting"]
    assert res.score == 0.0


def test_span_offsets_match_text():
    headline = "You Won't Believe What Happened Next"
    for res in score_headline(headline).values():
        for span in res.spans:
            assert headline[span.start : span.end] == span.text


def test_feature_vector_stable_names():
    feats = feature_vector("Some headline here")
    assert sorted(feats.keys()) == FEATURE_NAMES
    assert all(isinstance(v, float) for v in feats.values())
