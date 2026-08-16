"""Headline<->body "delivery gap": does the article deliver what the headline
promises?

Zero-shot NLI cross-encoder (default cross-encoder/nli-deberta-v3-xsmall,
~70 MB): premise = a window of body sentences, hypothesis = the headline.
The body is scored window-by-window and the best-entailed window wins —
a headline is "delivered" if ANY part of the article supports it.

  delivery_score = max over windows of P(entailment)
  gap_score      = 1 - delivery_score

This is an entailment claim about the ARTICLE, not about the world (ADR-0001):
a low delivery score means "the article doesn't support its own headline",
never "the story is false".

Roadmap: fine-tune on Webis clickbait17 (headline, body, truthMean) pairs.
"""

from __future__ import annotations

import functools
import re

from manipulens.config import load_params

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def body_windows(body: str, window_sents: int, max_windows: int) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(body) if s.strip()]
    if not sents:
        return []
    windows = [" ".join(sents[i : i + window_sents]) for i in range(0, len(sents), window_sents)]
    return windows[:max_windows]


@functools.lru_cache(maxsize=1)
def _load_nli():
    import torch  # noqa: F401  (import guard: optional heavy dep)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = load_params()["nli"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    # label order differs across NLI checkpoints — resolve from config
    id2label = {i: label.lower() for i, label in model.config.id2label.items()}
    entail_idx = next(i for i, label in id2label.items() if "entail" in label)
    contra_idx = next((i for i, label in id2label.items() if "contra" in label), None)
    return tokenizer, model, entail_idx, contra_idx


def nli_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def score_delivery(headline: str, body: str) -> dict:
    """Score how well the body delivers the headline's promise."""
    import torch

    params = load_params()["nli"]
    tokenizer, model, entail_idx, contra_idx = _load_nli()

    windows = body_windows(body, params["body_window_sents"], params["max_windows"])
    if not windows:
        return {"error": "empty body"}

    enc = tokenizer(
        windows,
        [headline] * len(windows),
        truncation=True,
        max_length=params["max_length"],
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)

    entail = probs[:, entail_idx]
    best = int(entail.argmax())
    delivery = float(entail[best])
    result = {
        "delivery_score": round(delivery, 3),
        "gap_score": round(1 - delivery, 3),
        "verdict": (
            "delivered"
            if delivery >= 0.6
            else "partially_supported"
            if delivery >= 0.3
            else "overpromises"
        ),
        "best_window": windows[best][:280],
        "n_windows": len(windows),
        "disclaimer": (
            "Measures whether the article supports its own headline — "
            "not whether the story is true."
        ),
    }
    if contra_idx is not None:
        result["max_contradiction"] = round(float(probs[:, contra_idx].max()), 3)
    return result
