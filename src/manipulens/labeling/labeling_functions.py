"""Weak-supervision labeling functions for the manipulation taxonomy.

Each labeling function (LF) maps a headline to a score in [0, 1] for one
taxonomy dimension AND returns the character spans that triggered it. The spans
serve double duty:

  1. weak labels / features for model training
  2. user-facing explanations ("highlighted trigger phrases") in the API

LFs are deliberately transparent (lexicons + regexes, versioned in git) — they
are an LLM-independent signal source and become CheckList probes for the
behavioral test suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources


@dataclass
class Span:
    start: int
    end: int
    text: str
    dimension: str


@dataclass
class DimensionResult:
    dimension: str
    score: float  # 0..1
    spans: list[Span] = field(default_factory=list)


DIMENSIONS = [
    "curiosity_gap",
    "outrage",
    "fear",
    "false_certainty",
    "emotional_framing",
    "sensational_formatting",
]

_LEXICON_FILES = {
    "curiosity_gap": "curiosity.txt",
    "outrage": "outrage.txt",
    "fear": "fear.txt",
    "false_certainty": "certainty.txt",
    "emotional_framing": "emotion.txt",
}


@lru_cache(maxsize=None)
def load_lexicon(dimension: str) -> tuple[str, ...]:
    fname = _LEXICON_FILES[dimension]
    text = (resources.files("manipulens.labeling") / "lexicons" / fname).read_text()
    terms = [ln.strip().lower() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    # longest-first so multi-word phrases win over substrings
    return tuple(sorted(terms, key=len, reverse=True))


@lru_cache(maxsize=None)
def _lexicon_pattern(dimension: str) -> re.Pattern[str]:
    terms = load_lexicon(dimension)
    alternation = "|".join(re.escape(t) for t in terms)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def _lexicon_lf(headline: str, dimension: str) -> DimensionResult:
    spans = [
        Span(m.start(), m.end(), m.group(0), dimension)
        for m in _lexicon_pattern(dimension).finditer(headline)
    ]
    # one hit = 0.6, two+ = 1.0 — mild vs. strong, mirroring the codebook's 0-2 scale
    score = 0.0 if not spans else (0.6 if len(spans) == 1 else 1.0)
    return DimensionResult(dimension, score, spans)


# --- sensational formatting: structural, not lexical ---

_ALLCAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
_MULTI_PUNCT_RE = re.compile(r"[!?]{2,}|!")
_LISTICLE_RE = re.compile(r"^\s*\d{1,3}\s+\w|\b\d{1,3}\s+(?:things|ways|reasons|facts|tricks|signs|photos|times)\b", re.IGNORECASE)
_NUMBER_TEASE_RE = re.compile(r"#\d+\b|\bnumber\s+\d+\b", re.IGNORECASE)

_STOP_ALLCAPS = {"CEO", "USA", "NASA", "FBI", "CIA", "GOP", "NFL", "NBA", "MLB", "CDC", "WHO", "EU", "UK", "US", "TV", "AI", "GDP", "IPO", "DNA", "PSA"}


def sensational_formatting_lf(headline: str) -> DimensionResult:
    dimension = "sensational_formatting"
    spans: list[Span] = []
    hits = 0.0

    for m in _ALLCAPS_RE.finditer(headline):
        if m.group(0) not in _STOP_ALLCAPS:
            spans.append(Span(m.start(), m.end(), m.group(0), dimension))
            hits += 1
    for m in _MULTI_PUNCT_RE.finditer(headline):
        weight = 1.0 if len(m.group(0)) > 1 else 0.5
        spans.append(Span(m.start(), m.end(), m.group(0), dimension))
        hits += weight
    m = _LISTICLE_RE.search(headline)
    if m:
        spans.append(Span(m.start(), m.end(), m.group(0), dimension))
        hits += 1
    m = _NUMBER_TEASE_RE.search(headline)
    if m:
        spans.append(Span(m.start(), m.end(), m.group(0), dimension))
        hits += 1

    score = min(1.0, hits * 0.5)
    return DimensionResult(dimension, score, spans)


# --- second-person curiosity boost ---

_SECOND_PERSON_RE = re.compile(r"\b(?:you|your|you're|youre)\b", re.IGNORECASE)
_FORWARD_REF_RE = re.compile(r"\b(?:this|these)\s+(?:one\s+)?\w+", re.IGNORECASE)


def score_headline(headline: str) -> dict[str, DimensionResult]:
    """Run all labeling functions on a headline. Returns dimension -> result."""
    results: dict[str, DimensionResult] = {}
    for dim in _LEXICON_FILES:
        results[dim] = _lexicon_lf(headline, dim)
    results["sensational_formatting"] = sensational_formatting_lf(headline)

    # curiosity boost: forward reference + second person with no lexicon hit yet
    cur = results["curiosity_gap"]
    if cur.score < 0.6 and _FORWARD_REF_RE.search(headline) and _SECOND_PERSON_RE.search(headline):
        cur.score = max(cur.score, 0.4)
    return results


def feature_vector(headline: str) -> dict[str, float]:
    """Numeric features for tabular models: LF scores + structural stats."""
    results = score_headline(headline)
    feats = {f"lf_{dim}": res.score for dim, res in results.items()}
    words = headline.split()
    feats["n_chars"] = float(len(headline))
    feats["n_words"] = float(len(words))
    feats["second_person"] = 1.0 if _SECOND_PERSON_RE.search(headline) else 0.0
    feats["starts_with_number"] = 1.0 if re.match(r"^\s*\d", headline) else 0.0
    feats["question_mark"] = 1.0 if "?" in headline else 0.0
    feats["exclamation"] = 1.0 if "!" in headline else 0.0
    caps_words = [w for w in words if w.isupper() and len(w) >= 3 and w not in _STOP_ALLCAPS]
    feats["allcaps_ratio"] = len(caps_words) / max(1, len(words))
    return feats


FEATURE_NAMES = sorted(feature_vector("placeholder headline").keys())
