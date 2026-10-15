import pytest

from edgeveil.evaluation.bootstrap import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    score_with_intervals,
)

PERFECT_GOLD = [["B-PERSON", "I-PERSON", "O"], ["O", "B-EMAIL"], ["O", "O"]]
HALF_RIGHT = [["B-PERSON", "I-PERSON", "O"], ["O", "O"], ["O", "O"]]


def test_constants_are_the_agreed_values() -> None:
    assert CONFIDENCE_LEVEL == 0.95
    assert BOOTSTRAP_RESAMPLES >= 1000
    assert isinstance(BOOTSTRAP_SEED, int)


def test_perfect_predictions_have_a_degenerate_interval() -> None:
    scores = score_with_intervals(PERFECT_GOLD, PERFECT_GOLD, resamples=200)
    person = scores["PERSON"]
    assert person.f1.value == person.f1.low == person.f1.high == pytest.approx(1.0)
    assert person.support == 1


def test_interval_brackets_the_point_estimate() -> None:
    gold = PERFECT_GOLD * 20
    predicted = HALF_RIGHT * 20
    overall = score_with_intervals(gold, predicted, resamples=300)["overall"]
    assert overall.recall.low <= overall.recall.value <= overall.recall.high
    assert overall.recall.value == pytest.approx(0.5)
    assert overall.recall.low < overall.recall.high


def test_same_seed_gives_the_same_interval() -> None:
    gold = PERFECT_GOLD * 10
    predicted = HALF_RIGHT * 10
    first = score_with_intervals(gold, predicted, resamples=100, seed=7)
    second = score_with_intervals(gold, predicted, resamples=100, seed=7)
    assert first == second


def test_types_missing_from_gold_and_predictions_are_absent() -> None:
    scores = score_with_intervals(PERFECT_GOLD, PERFECT_GOLD, resamples=50)
    assert "SSN" not in scores
