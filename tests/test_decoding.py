import numpy as np
import pytest

from edgeveil.detector.alignment import SPECIAL_TOKEN_OFFSET
from edgeveil.detector.decoding import (
    merge_window_spans,
    thresholded_label_ids,
    window_spans,
)
from edgeveil.detector.labels import BIO_LABELS, LABEL_TO_ID
from edgeveil.spans import EntityType, Span, SpanSource

UNIFORM = dict.fromkeys(EntityType, 0.5)


def probabilities(*rows: dict[str, float]) -> np.ndarray:
    table = np.zeros((len(rows), len(BIO_LABELS)))
    for index, row in enumerate(rows):
        for label, value in row.items():
            table[index, LABEL_TO_ID[label]] = value
        table[index, LABEL_TO_ID["O"]] += 1.0 - sum(row.values())
    return table


def span(start: int, end: int, label: EntityType, score: float) -> Span:
    return Span(start, end, label, score, SpanSource.MODEL)


def test_entity_mass_combines_begin_and_inside() -> None:
    table = probabilities({"B-EMAIL": 0.3, "I-EMAIL": 0.3})
    assert thresholded_label_ids(table, UNIFORM) == [LABEL_TO_ID["B-EMAIL"]]


def test_mass_below_the_type_threshold_becomes_outside() -> None:
    table = probabilities({"B-EMAIL": 0.6})
    thresholds = {**UNIFORM, EntityType.EMAIL: 0.7}
    assert thresholded_label_ids(table, thresholds) == [LABEL_TO_ID["O"]]


def test_inside_is_chosen_when_it_outweighs_begin() -> None:
    table = probabilities({"B-PHONE": 0.9}, {"B-PHONE": 0.2, "I-PHONE": 0.7})
    assert thresholded_label_ids(table, UNIFORM) == [
        LABEL_TO_ID["B-PHONE"],
        LABEL_TO_ID["I-PHONE"],
    ]


def test_window_spans_carry_the_mean_entity_mass_as_score() -> None:
    table = probabilities({}, {"B-PERSON": 0.8}, {"I-PERSON": 0.6}, {})
    offsets = [SPECIAL_TOKEN_OFFSET, (0, 3), (4, 9), SPECIAL_TOKEN_OFFSET]
    [found] = window_spans(table, offsets, UNIFORM)
    assert (found.start, found.end, found.label) == (0, 9, EntityType.PERSON)
    assert found.score == pytest.approx(0.7)
    assert found.source is SpanSource.MODEL


def test_overlapping_spans_of_one_type_merge_across_windows() -> None:
    merged = merge_window_spans(
        [span(0, 9, EntityType.PERSON, 0.7), span(4, 15, EntityType.PERSON, 0.9)]
    )
    assert merged == [span(0, 15, EntityType.PERSON, 0.9)]


def test_identical_spans_from_two_windows_collapse() -> None:
    first = span(5, 20, EntityType.EMAIL, 0.8)
    assert merge_window_spans([first, span(5, 20, EntityType.EMAIL, 0.6)]) == [first]


def test_overlapping_spans_of_different_types_keep_the_higher_score() -> None:
    merged = merge_window_spans(
        [span(0, 10, EntityType.PHONE, 0.6), span(5, 12, EntityType.GOV_ID, 0.8)]
    )
    assert merged == [span(5, 12, EntityType.GOV_ID, 0.8)]


def test_touching_spans_stay_separate() -> None:
    spans = [span(0, 3, EntityType.PERSON, 0.9), span(3, 6, EntityType.PERSON, 0.9)]
    assert merge_window_spans(spans) == spans
