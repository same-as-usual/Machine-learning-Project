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
