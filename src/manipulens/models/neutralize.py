"""Political-entity masking: neutrality by construction.

Technique detection must be invariant to WHO a headline is about. Instead of
hoping the training data is balanced, we replace political entities and
identity terms with a shared placeholder token BEFORE the learned text
representation (TF-IDF / future transformer tokenization). Listed entities
therefore carry exactly zero signal.

The behavioral suite (tests/behavioral/) still verifies invariance end-to-end,
and the Phase 7 neutrality audit covers entities NOT in the lexicon.

The lexicon lives in labeling/lexicons/political_entities.txt — transparent,
versioned, and auditable like every other lexicon.
"""

from __future__ import annotations

import re
from functools import cache
from importlib import resources

MASK_TOKEN = "entitytoken"


@cache
def _pattern() -> re.Pattern[str]:
    text = (
        resources.files("manipulens.labeling") / "lexicons" / "political_entities.txt"
    ).read_text()
    terms = [
        ln.strip().lower() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")
    ]
    terms = sorted(terms, key=len, reverse=True)  # longest-first: phrases beat substrings
    alternation = "|".join(re.escape(t) for t in terms)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def mask_entities(text: str) -> str:
    """Replace political entities with MASK_TOKEN."""
    return _pattern().sub(MASK_TOKEN, text)
