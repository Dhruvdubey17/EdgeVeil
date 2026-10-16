import numpy as np
import pytest

from edgeveil.detector.alignment import SPECIAL_TOKEN_OFFSET
from edgeveil.detector.decoding import (
    candidate_label_ids,
    merge_window_spans,
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
    assert candidate_label_ids(table) == [LABEL_TO_ID["B-EMAIL"]]


def test_token_is_outside_when_outside_outweighs_every_type() -> None:
    table = probabilities({"B-EMAIL": 0.3, "B-PHONE": 0.2})
    assert candidate_label_ids(table) == [LABEL_TO_ID["O"]]


def test_span_below_the_type_threshold_is_dropped() -> None:
    table = probabilities({"B-EMAIL": 0.6})
    thresholds = {**UNIFORM, EntityType.EMAIL: 0.7}
    assert window_spans(table, [(0, 5)], thresholds) == []


def test_one_uncertain_inner_token_does_not_split_a_span() -> None:
    table = probabilities(
        {"B-CREDENTIAL": 0.99}, {"I-CREDENTIAL": 0.6}, {"I-CREDENTIAL": 0.99}
    )
    thresholds = {**UNIFORM, EntityType.CREDENTIAL: 0.8}
    [found] = window_spans(table, [(0, 4), (4, 8), (8, 12)], thresholds)
    assert (found.start, found.end, found.label) == (0, 12, EntityType.CREDENTIAL)
    assert found.score == pytest.approx(0.86)


def test_inside_is_chosen_when_it_outweighs_begin() -> None:
    table = probabilities({"B-PHONE": 0.9}, {"B-PHONE": 0.2, "I-PHONE": 0.7})
    assert candidate_label_ids(table) == [
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


TOKENS = [(0, 4), (5, 9), (10, 15), (16, 20)]


def test_overlapping_spans_of_one_type_merge_across_windows() -> None:
    merged = merge_window_spans(
        [span(0, 9, EntityType.PERSON, 0.7), span(5, 15, EntityType.PERSON, 0.9)],
        TOKENS,
    )
    assert merged == [span(0, 15, EntityType.PERSON, 0.9)]


def test_identical_spans_from_two_windows_collapse() -> None:
    first = span(5, 20, EntityType.EMAIL, 0.8)
    merged = merge_window_spans([first, span(5, 20, EntityType.EMAIL, 0.6)], TOKENS)
    assert merged == [first]


def test_the_higher_score_wins_an_overlap_and_the_rest_stays_masked() -> None:
    merged = merge_window_spans(
        [span(0, 15, EntityType.PHONE, 0.6), span(5, 20, EntityType.GOV_ID, 0.8)],
        TOKENS,
    )
    assert merged == [
        span(0, 4, EntityType.PHONE, 0.6),
        span(5, 20, EntityType.GOV_ID, 0.8),
    ]


def test_touching_spans_of_one_type_merge_and_of_two_types_stay_separate() -> None:
    same = [span(0, 4, EntityType.PERSON, 0.6), span(4, 9, EntityType.PERSON, 0.9)]
    assert merge_window_spans(same, TOKENS) == [span(0, 9, EntityType.PERSON, 0.9)]
    mixed = [span(0, 4, EntityType.PERSON, 0.9), span(4, 9, EntityType.EMAIL, 0.9)]
    assert merge_window_spans(mixed, TOKENS) == mixed


def test_touching_tokens_of_one_type_form_one_span_before_the_threshold() -> None:
    table = probabilities({"B-EMAIL": 0.9}, {"B-EMAIL": 0.9}, {"B-EMAIL": 0.9})
    offsets = [(0, 5), (5, 6), (6, 20)]
    [found] = window_spans(table, offsets, UNIFORM)
    assert (found.start, found.end, found.label) == (0, 20, EntityType.EMAIL)
