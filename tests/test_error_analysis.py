import math

import pytest

from edgeveil.evaluation.errors import (
    ErrorKind,
    ErrorSide,
    detection_errors,
    error_counts,
)
from edgeveil.evaluation.secondary import match_counts, secondary_scores
from edgeveil.spans import EntityType, GoldSpan, Span, SpanSource

PERSON = EntityType.PERSON
EMAIL = EntityType.EMAIL


def emitted(start: int, end: int, label: EntityType) -> Span:
    return Span(start, end, label, 0.9, SpanSource.MODEL)


def kinds(
    gold: list[GoldSpan], predicted: list[GoldSpan]
) -> list[tuple[ErrorSide, ErrorKind]]:
    return [(error.side, error.kind) for error in detection_errors([gold], [predicted])]


def test_exact_matches_are_not_errors() -> None:
    span = GoldSpan(0, 9, PERSON)
    assert kinds([span], [span]) == []


def test_a_gold_span_with_no_overlap_is_missed_and_an_extra_span_is_spurious() -> None:
    assert kinds([GoldSpan(0, 9, PERSON)], [GoldSpan(20, 30, EMAIL)]) == [
        (ErrorSide.MISS, ErrorKind.MISSED),
        (ErrorSide.FALSE_POSITIVE, ErrorKind.SPURIOUS),
    ]


def test_same_type_overlap_is_a_boundary_error_on_both_sides() -> None:
    assert kinds([GoldSpan(0, 9, PERSON)], [GoldSpan(0, 3, PERSON)]) == [
        (ErrorSide.MISS, ErrorKind.BOUNDARY),
        (ErrorSide.FALSE_POSITIVE, ErrorKind.BOUNDARY),
    ]


def test_other_type_overlap_is_a_wrong_type_error() -> None:
    assert kinds([GoldSpan(0, 9, PERSON)], [GoldSpan(0, 9, EMAIL)]) == [
        (ErrorSide.MISS, ErrorKind.WRONG_TYPE),
        (ErrorSide.FALSE_POSITIVE, ErrorKind.WRONG_TYPE),
    ]


def test_same_type_overlap_wins_over_other_type_overlap() -> None:
    gold = [GoldSpan(0, 9, PERSON)]
    predicted = [GoldSpan(0, 3, EMAIL), GoldSpan(4, 9, PERSON)]
    errors = detection_errors([gold], [predicted])
    assert errors[0].kind is ErrorKind.BOUNDARY
    assert errors[0].counterpart == predicted[1]


def test_counts_list_every_type_and_kind_with_zeros() -> None:
    errors = detection_errors([[GoldSpan(0, 9, PERSON)]], [[]])
    counts = error_counts(errors)
    assert counts["misses"]["PERSON"] == {"missed": 1, "boundary": 0, "wrong_type": 0}
    assert counts["false_positives"]["SSN"] == {
        "spurious": 0,
        "boundary": 0,
        "wrong_type": 0,
    }


def test_overlap_scores_forgive_boundaries_and_char_recall_ignores_type() -> None:
    gold = [[GoldSpan(0, 10, PERSON), GoldSpan(20, 30, EMAIL)]]
    predicted_raw = [[emitted(0, 4, PERSON), emitted(20, 25, PERSON)]]
    predicted = [
        [GoldSpan(span.start, span.end, span.label) for span in predicted_raw[0]]
    ]
    scores = secondary_scores(match_counts(gold, predicted, gold, predicted_raw), 20)
    assert scores["PERSON"].overlap_recall.value == 1.0
    assert scores["PERSON"].overlap_precision.value == 0.5
    assert scores["EMAIL"].overlap_recall.value == 0.0
    assert scores["overall"].char_recall.value == pytest.approx(9 / 20)
    assert scores["EMAIL"].char_recall.value == pytest.approx(0.5)


def test_types_without_spans_are_absent_and_empty_ratios_are_nan() -> None:
    gold = [[GoldSpan(0, 10, PERSON)], []]
    scores = secondary_scores(match_counts(gold, [[], []], gold, [[], []]), 20)
    assert "SSN" not in scores
    assert math.isnan(scores["PERSON"].overlap_precision.value)
    assert scores["PERSON"].char_recall.value == 0.0
