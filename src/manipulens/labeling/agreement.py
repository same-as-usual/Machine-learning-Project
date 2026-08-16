"""Krippendorff's alpha for annotation reliability.

Supports nominal and ordinal/interval metrics on the codebook's 0-2 scales.
Used to enforce the pre-committed reliability policy: dimensions with
ordinal alpha < 0.6 on the gold set are merged or dropped (codebook.md).

Implementation follows Krippendorff (2011), coincidence-matrix formulation.
Handles missing annotations (None / NaN).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Value = float | int | None


def _coincidence_matrix(
    units: list[list[float]],
) -> tuple[dict[tuple[float, float], float], dict[float, float], float]:
    coincidences: dict[tuple[float, float], float] = {}
    totals: dict[float, float] = {}
    n_total = 0.0
    for values in units:
        m = len(values)
        if m < 2:
            continue
        for i, vi in enumerate(values):
            for j, vj in enumerate(values):
                if i == j:
                    continue
                coincidences[(vi, vj)] = coincidences.get((vi, vj), 0.0) + 1.0 / (m - 1)
        for v in values:
            totals[v] = totals.get(v, 0.0) + 1.0
            n_total += 1.0
    return coincidences, totals, n_total


def _delta_nominal(a: float, b: float) -> float:
    return 0.0 if a == b else 1.0


def _delta_interval(a: float, b: float) -> float:
    return (a - b) ** 2


def krippendorff_alpha(
    annotations: Sequence[Sequence[Value]],
    metric: str = "interval",
) -> float:
    """annotations: one inner sequence per unit (headline), containing each
    annotator's value for that unit (None for missing).

    metric: 'nominal' or 'interval' (use 'interval' for the ordinal 0-2 scales;
    with equidistant ordinal categories the two coincide).
    """
    delta = _delta_nominal if metric == "nominal" else _delta_interval
    units = [
        [float(v) for v in unit if v is not None and not (isinstance(v, float) and math.isnan(v))]
        for unit in annotations
    ]
    units = [u for u in units if len(u) >= 2]
    if not units:
        raise ValueError("need at least one unit with >= 2 annotations")

    coincidences, totals, n = _coincidence_matrix(units)
    values = sorted(totals)
    if len(values) == 1:
        return 1.0  # perfect agreement, single category

    d_observed = sum(coincidences.get((a, b), 0.0) * delta(a, b) for a in values for b in values)
    d_expected = sum(
        totals[a] * totals[b] * delta(a, b)
        for a in values
        for b in values
        if a != b or delta(a, b) != 0
    ) / (n - 1)
    if d_expected == 0:
        return 1.0
    return 1.0 - d_observed / d_expected


def agreement_report(
    annotator_a: Sequence[Value],
    annotator_b: Sequence[Value],
    metric: str = "interval",
) -> dict[str, float]:
    """Two-annotator convenience wrapper."""
    units = list(zip(annotator_a, annotator_b, strict=True))
    alpha = krippendorff_alpha(units, metric=metric)
    paired = [(a, b) for a, b in units if a is not None and b is not None]
    exact = sum(1 for a, b in paired if a == b) / max(1, len(paired))
    return {"alpha": alpha, "exact_agreement": exact, "n_units": float(len(paired))}
