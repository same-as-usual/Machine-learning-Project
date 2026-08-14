"""ManipuLens scoring API.

    POST /score        {"headline": "..."}          -> technique scores + spans
    POST /score_batch  {"headlines": ["...", ...]}  -> list of results
    POST /feedback     user corrections queued for the retraining loop
    GET  /health

Run: uvicorn manipulens.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from manipulens import __version__
from manipulens.api.inference import get_model, score_one
from manipulens.config import REPO_ROOT

app = FastAPI(
    title="ManipuLens",
    version=__version__,
    description="Detects manipulation techniques in news headlines — not truth.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo; restrict for production
    allow_methods=["*"],
    allow_headers=["*"],
)

FEEDBACK_FILE = REPO_ROOT / "data" / "feedback" / "feedback.jsonl"

MAX_HEADLINE_LEN = 500
MAX_BATCH = 100


class ScoreRequest(BaseModel):
    headline: str = Field(min_length=1, max_length=MAX_HEADLINE_LEN)


class BatchScoreRequest(BaseModel):
    headlines: list[str] = Field(min_length=1, max_length=MAX_BATCH)


class FeedbackRequest(BaseModel):
    headline: str = Field(min_length=1, max_length=MAX_HEADLINE_LEN)
    model_score: float | None = None
    user_verdict: str = Field(pattern="^(agree|too_high|too_low)$")
    comment: str | None = Field(default=None, max_length=1000)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": get_model() is not None,
    }


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    return score_one(req.headline)


@app.post("/score_batch")
def score_batch(req: BatchScoreRequest) -> dict:
    for h in req.headlines:
        if len(h) > MAX_HEADLINE_LEN:
            raise HTTPException(422, detail="headline too long")
    return {"results": [score_one(h) for h in req.headlines]}


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """Append user corrections to a JSONL queue — the raw material for the
    review -> relabel -> retrain loop (see docs/PLAN.md Phase 6)."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = req.model_dump() | {"ts": time.time()}
    with FEEDBACK_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "queued"}


_STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_INDEX)


@app.get("/info")
def info() -> dict:
    return {
        "service": "ManipuLens",
        "docs": "/docs",
        "disclaimer": "Detects manipulation techniques, not truth.",
    }


def _feedback_count() -> int:  # used in tests
    path = Path(FEEDBACK_FILE)
    if not path.exists():
        return 0
    return sum(1 for _ in path.open())
