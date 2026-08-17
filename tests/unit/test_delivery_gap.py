from manipulens.models.delivery_gap import body_windows


def test_body_windows_chunks_sentences():
    body = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine."
    windows = body_windows(body, window_sents=4, max_windows=6)
    assert len(windows) == 3
    assert windows[0] == "One. Two. Three. Four."


def test_body_windows_empty():
    assert body_windows("", 4, 6) == []
    assert body_windows("   ", 4, 6) == []


def test_body_windows_respects_max():
    body = ". ".join(f"Sentence {i}" for i in range(100)) + "."
    windows = body_windows(body, window_sents=2, max_windows=6)
    assert len(windows) == 6


def test_verdict_combines_entailment_and_contradiction():
    from manipulens.models.delivery_gap import _verdict

    # wire-style headline: body directly entails it
    assert _verdict(0.95, 0.01) == "delivered"
    # overpromising headline: body undercuts its own claim
    assert _verdict(0.002, 0.65) == "overpromises"
    # faithful paraphrase: near-zero entailment AND near-zero contradiction
    # must NOT be called overpromising (the curcumin probe pair regression)
    assert _verdict(0.003, 0.002) == "not_directly_supported"
    # moderate entailment, low contradiction
    assert _verdict(0.4, 0.05) == "partially_supported"
    # low entailment with meaningful contradiction leans overpromising
    assert _verdict(0.1, 0.2) == "overpromises"
