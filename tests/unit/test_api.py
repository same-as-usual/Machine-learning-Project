from fastapi.testclient import TestClient

from manipulens.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_score_returns_techniques_and_disclaimer():
    resp = client.post("/score", json={"headline": "You Won't Believe What This Senator Said Next"})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["manipulation_score"] <= 1.0
    assert "curiosity_gap" in body["techniques"]
    assert "truth" not in str(body["detected_techniques"]).lower()  # ADR-0001
    assert "not factual accuracy" in body["disclaimer"]


def test_score_rejects_empty():
    assert client.post("/score", json={"headline": ""}).status_code == 422


def test_score_rejects_overlong():
    assert client.post("/score", json={"headline": "x" * 1000}).status_code == 422


def test_batch():
    resp = client.post(
        "/score_batch",
        json={"headlines": ["Fed holds rates steady", "27 INSANE tricks!!"]},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert results[1]["manipulation_score"] >= results[0]["manipulation_score"]


def test_feedback_queued(tmp_path, monkeypatch):
    import manipulens.api.main as m

    monkeypatch.setattr(m, "FEEDBACK_FILE", tmp_path / "fb.jsonl")
    resp = client.post(
        "/feedback",
        json={"headline": "Some headline", "user_verdict": "too_high"},
    )
    assert resp.status_code == 200
    assert (tmp_path / "fb.jsonl").read_text().count("\n") == 1


def test_feedback_rejects_bad_verdict():
    resp = client.post("/feedback", json={"headline": "x y z", "user_verdict": "nonsense"})
    assert resp.status_code == 422
