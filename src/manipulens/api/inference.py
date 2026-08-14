"""Model loading + scoring logic shared by the API.

Combines:
  - calibrated clickbait-intensity probability from the trained baseline
  - per-dimension technique scores + trigger spans from labeling functions

Output language follows ADR-0001: techniques, never truth verdicts.
"""

from __future__ import annotations

import functools
from typing import Any

import joblib

from manipulens.config import artifacts_dir
from manipulens.labeling.labeling_functions import score_headline

MODEL_FILE = "baseline.joblib"

_INTENSITY_BANDS = [
    (0.35, "low"),
    (0.65, "moderate"),
    (1.01, "high"),
]


@functools.lru_cache(maxsize=1)
def get_model() -> Any | None:
    path = artifacts_dir("models") / MODEL_FILE
    if not path.exists():
        return None
    return joblib.load(path)


def intensity_band(p: float) -> str:
    for cutoff, name in _INTENSITY_BANDS:
        if p < cutoff:
            return name
    return "high"


def score_one(headline: str) -> dict[str, Any]:
    model = get_model()
    if model is not None:
        p = float(model.predict_proba([headline], calibrated=True)[0])
        model_name = "baseline-lgbm-calibrated"
    else:  # degrade gracefully to rules if no artifact is present
        results_only = score_headline(headline)
        p = min(1.0, sum(r.score for r in results_only.values()) / len(results_only) * 2)
        model_name = "rules-fallback"

    dim_results = score_headline(headline)
    techniques = {
        dim: {
            "score": round(res.score, 3),
            "spans": [{"start": s.start, "end": s.end, "text": s.text} for s in res.spans],
        }
        for dim, res in dim_results.items()
    }
    detected = sorted(
        (dim for dim, res in dim_results.items() if res.score >= 0.6),
        key=lambda d: -dim_results[d].score,
    )
    return {
        "headline": headline,
        "manipulation_score": round(p, 3),
        "intensity": intensity_band(p),
        "techniques": techniques,
        "detected_techniques": detected,
        "model": model_name,
        "disclaimer": "Scores describe rhetorical techniques, not factual accuracy.",
    }
