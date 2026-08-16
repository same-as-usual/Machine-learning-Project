import pytest

from manipulens.labeling.agreement import agreement_report, krippendorff_alpha


def test_perfect_agreement():
    units = [(0, 0), (1, 1), (2, 2), (1, 1), (0, 0)]
    assert krippendorff_alpha(units, metric="interval") == pytest.approx(1.0)


def test_known_value_nominal():
    # Krippendorff (2011) worked binary example, 2 observers, 10 units: alpha ≈ 0.095
    a = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    b = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    alpha = krippendorff_alpha(list(zip(a, b, strict=True)), metric="nominal")
    assert alpha == pytest.approx(0.095, abs=0.005)


def test_disagreement_lower_than_agreement():
    agree = [(0, 0), (1, 1), (2, 2), (0, 0), (2, 2), (1, 1)]
    disagree = [(0, 2), (1, 0), (2, 0), (0, 1), (2, 1), (1, 2)]
    assert krippendorff_alpha(agree) > krippendorff_alpha(disagree)


def test_interval_penalizes_distance():
    near = [(0, 1), (1, 2), (1, 0), (2, 1), (0, 1), (2, 1)]
    far = [(0, 2), (2, 0), (0, 2), (2, 0), (0, 2), (2, 0)]
    assert krippendorff_alpha(near, "interval") > krippendorff_alpha(far, "interval")


def test_missing_values_handled():
    units = [(0, 0), (1, None), (2, 2), (None, 1), (1, 1)]
    alpha = krippendorff_alpha(units)
    assert -1.0 <= alpha <= 1.0


def test_report_wrapper():
    rep = agreement_report([0, 1, 2, 1, 0, 2], [0, 1, 2, 2, 0, 2])
    assert rep["n_units"] == 6
    assert rep["exact_agreement"] == pytest.approx(5 / 6)
    assert 0 < rep["alpha"] <= 1


def test_requires_pairs():
    with pytest.raises(ValueError):
        krippendorff_alpha([(1, None), (None, 2)])
