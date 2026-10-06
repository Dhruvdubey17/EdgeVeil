import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from edgeveil.spans import (
    MAX_SCORE,
    MIN_SCORE,
    EntityType,
    InvalidSpanError,
    Span,
    SpanSource,
)

offsets = st.integers(min_value=0, max_value=10_000)
lengths = st.integers(min_value=1, max_value=10_000)
valid_scores = st.floats(min_value=MIN_SCORE, max_value=MAX_SCORE)


def make_span(start: int, end: int, score: float = MAX_SCORE) -> Span:
    return Span(
        start=start,
        end=end,
        label=EntityType.EMAIL,
        score=score,
        source=SpanSource.MODEL,
    )


@given(start=offsets, length=lengths, score=valid_scores)
def test_accepts_any_nonempty_span_with_score_in_unit_interval(
    start: int, length: int, score: float
) -> None:
    span = make_span(start, start + length, score)
    assert (span.start, span.end, span.score) == (start, start + length, score)


def test_rejects_negative_start() -> None:
    with pytest.raises(InvalidSpanError):
        make_span(-1, 4)


@pytest.mark.parametrize("end", [3, 2])
def test_rejects_empty_or_reversed_span(end: int) -> None:
    with pytest.raises(InvalidSpanError):
        make_span(3, end)


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan")])
def test_rejects_score_outside_unit_interval(score: float) -> None:
    with pytest.raises(InvalidSpanError):
        make_span(0, 4, score)


@pytest.mark.parametrize("field", [field.name for field in dataclasses.fields(Span)])
def test_is_frozen(field: str) -> None:
    span = make_span(0, 4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(span, field, 1)


def test_is_slotted() -> None:
    assert not hasattr(make_span(0, 4), "__dict__")


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ((0, 5), (3, 8), True),
        ((3, 8), (0, 5), True),
        ((0, 10), (2, 4), True),
        ((0, 5), (5, 8), False),
        ((5, 8), (0, 5), False),
    ],
)
def test_overlaps_uses_exclusive_end(
    first: tuple[int, int], second: tuple[int, int], expected: bool
) -> None:
    assert make_span(*first).overlaps(make_span(*second)) is expected


def test_entity_type_values_match_placeholder_names() -> None:
    assert all(member.value == member.name for member in EntityType)
