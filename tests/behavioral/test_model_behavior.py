"""CheckList-style behavioral tests — the model promotion gate.

These run against the trained artifact (models/artifacts/baseline.joblib) and
encode expectations a manipulation-technique model MUST satisfy:

  - Invariance: swapping named political entities must not move scores.
  - Directional: adding clickbait devices must raise the score.
  - Sanity: wire-style headlines score below archetypal clickbait.

If the artifact is missing the suite skips (unit CI). The `behavioral-gate` CI
job trains on sample data first, so these always gate promotion on main.
"""

import pytest

from manipulens.api.inference import get_model

pytestmark = pytest.mark.behavioral

ENTITY_SWAP_TOLERANCE = 0.10  # max allowed |Δscore| under political entity swap
DIRECTIONAL_MIN_LIFT = 0.05  # min score increase when adding a clickbait device


@pytest.fixture(scope="module")
def model():
    m = get_model()
    if m is None:
        pytest.skip("no trained artifact; run `make data train` first")
    return m


def _score(model, headline: str) -> float:
    return float(model.predict_proba([headline], calibrated=True)[0])


# --- Invariance: political neutrality ---

ENTITY_PAIRS = [
    ("Biden", "Trump"),
    ("Democrats", "Republicans"),
    ("the left", "the right"),
    ("liberal", "conservative"),
]

TEMPLATES = [
    "{e} announces new economic policy plan",
    "{e} slams critics in fiery speech",
    "You won't believe what {e} said next",
    "{e} responds to questions about the budget",
]


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("pair", ENTITY_PAIRS, ids=lambda p: f"{p[0]}-vs-{p[1]}")
def test_entity_swap_invariance(model, template, pair):
    a, b = pair
    delta = abs(_score(model, template.format(e=a)) - _score(model, template.format(e=b)))
    assert delta <= ENTITY_SWAP_TOLERANCE, (
        f"score moved {delta:.3f} swapping {a!r}->{b!r} in {template!r}"
    )


# --- Directional: adding manipulation devices must raise the score ---

NEUTRAL_BASES = [
    "Senate passes infrastructure bill in 69-30 vote",
    "Federal Reserve holds interest rates steady",
    "City council approves 2026 budget after public hearing",
]


@pytest.mark.parametrize("base", NEUTRAL_BASES)
def test_clickbait_prefix_raises_score(model, base):
    hyped = f"You won't believe this: {base}"
    assert _score(model, hyped) >= _score(model, base) + DIRECTIONAL_MIN_LIFT


@pytest.mark.parametrize("base", NEUTRAL_BASES)
def test_shock_framing_raises_score(model, base):
    hyped = f"SHOCKING: {base} — what happened next will stun you"
    assert _score(model, hyped) >= _score(model, base) + DIRECTIONAL_MIN_LIFT


# --- Sanity ordering ---

def test_archetypes_ordered(model):
    wire = "Court upholds ruling in state redistricting case"
    bait = "You Won't Believe What This Judge Did Next — #3 Will SHOCK You!!"
    assert _score(model, bait) > _score(model, wire)


def test_scores_are_probabilities(model):
    for h in NEUTRAL_BASES:
        s = _score(model, h)
        assert 0.0 <= s <= 1.0
