from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edgeveil.detector.alignment import SPECIAL_TOKEN_OFFSET
from edgeveil.detector.decoding import (
    candidate_label_ids,
    join_touching,
    merge_window_spans,
    touches_same_type,
    window_spans,
)
from edgeveil.detector.labels import BIO_LABELS, LABEL_TO_ID
from edgeveil.spans import EntityType, GoldSpan, Span, SpanSource

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


def test_touching_spans_of_one_type_merge_and_of_two_types_stay_separate() -> None:
    same = [span(0, 3, EntityType.PERSON, 0.6), span(3, 6, EntityType.PERSON, 0.9)]
    assert merge_window_spans(same) == [span(0, 6, EntityType.PERSON, 0.9)]
    mixed = [span(0, 3, EntityType.PERSON, 0.9), span(3, 6, EntityType.EMAIL, 0.9)]
    assert merge_window_spans(mixed) == mixed


def test_touching_tokens_of_one_type_form_one_span_before_the_threshold() -> None:
    table = probabilities({"B-EMAIL": 0.9}, {"B-EMAIL": 0.9}, {"B-EMAIL": 0.9})
    offsets = [(0, 5), (5, 6), (6, 20)]
    [found] = window_spans(table, offsets, UNIFORM)
    assert (found.start, found.end, found.label) == (0, 20, EntityType.EMAIL)


LABELS = st.sampled_from([EntityType.PERSON, EntityType.EMAIL])


@st.composite
def ordered_spans(draw: st.DrawFn) -> list[GoldSpan]:
    gaps = draw(st.lists(st.integers(0, 2), min_size=1, max_size=12))
    lengths = draw(st.lists(st.integers(1, 4), min_size=len(gaps), max_size=len(gaps)))
    spans: list[GoldSpan] = []
    cursor = 0
    for gap, length in zip(gaps, lengths, strict=True):
        cursor += gap
        spans.append(GoldSpan(cursor, cursor + length, draw(LABELS)))
        cursor += length
    return spans


def covered(spans: list[GoldSpan] | list[Span], label: EntityType) -> set[int]:
    return {
        index
        for item in spans
        if item.label is label
        for index in range(item.start, item.end)
    }


@given(ordered_spans())
def test_joining_leaves_no_touching_pair_and_keeps_coverage(
    spans: list[GoldSpan],
) -> None:
    joined = join_touching(spans)
    assert not any(touches_same_type(left, right) for left, right in pairwise(joined))
    for label in (EntityType.PERSON, EntityType.EMAIL):
        assert covered(joined, label) == covered(spans, label)


@given(ordered_spans())
def test_window_merge_joins_touching_spans_of_one_type(spans: list[GoldSpan]) -> None:
    model_spans = [span(item.start, item.end, item.label, 0.9) for item in spans]
    merged = merge_window_spans(model_spans)
    assert not any(touches_same_type(left, right) for left, right in pairwise(merged))
    for label in (EntityType.PERSON, EntityType.EMAIL):
        assert covered(merged, label) == covered(model_spans, label)
