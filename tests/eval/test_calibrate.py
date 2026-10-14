import pytest

from src.eval.calibrate import Point, calibrate


def points(answerable: list[float], refuse: list[float]) -> list[Point]:
    return [Point(f"a{i}", "answer", c) for i, c in enumerate(answerable)] + [
        Point(f"r{i}", "refuse", c) for i, c in enumerate(refuse)
    ]


def test_separable_scores_put_the_threshold_in_the_middle_of_the_gap():
    cal = calibrate(points([0.6, 0.8, 0.9, 0.95], [0.001, 0.02, 0.1]), max_false_refusal=0.05)
    # 5% of 4 answerable = 0 may be refused: the cap is 0.6, the gap is (0.1, 0.6).
    assert cal.threshold == pytest.approx(0.35)
    assert cal.false_refusal_rate == 0 and cal.catch_rate == 1
    assert cal.margin_below == pytest.approx(0.25) and cal.margin_above == pytest.approx(0.25)
    assert not cal.missed


def test_overlap_allows_the_capped_share_of_false_refusals():
    answerable = [0.05, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95]
    cal = calibrate(points(answerable, [0.01, 0.2, 0.4, 0.97]), max_false_refusal=0.1)
    # 10% of 10 = 1 may be refused (0.05); 0.3 must pass: the gap is (0.2, 0.3).
    assert cal.threshold == pytest.approx(0.25)
    assert [p.confidence for p in cal.refused_answerable] == [0.05]
    assert [p.confidence for p in cal.caught] == [0.01, 0.2]
    assert [p.confidence for p in cal.missed] == [0.4, 0.97]  # left to the model
    assert cal.catch_rate == 0.5


def test_too_few_unanswerable_turns_warn():
    cal = calibrate(points([0.5, 0.9], [0.01]), max_false_refusal=0.05)
    assert any("unanswerable" in w for w in cal.warnings)


def test_no_answerable_turns_raise():
    with pytest.raises(ValueError):
        calibrate(points([], [0.1]), max_false_refusal=0.05)
