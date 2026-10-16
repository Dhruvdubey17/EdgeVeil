import re
from collections.abc import Sequence
from dataclasses import replace

from edgeveil.spans import EntityType, Span
from edgeveil.validators.finders import (
    EMAIL,
    digits_of,
    has_phone_digit_count,
    is_valid_card_number,
)

OVERRIDING_VALIDATOR_TYPES = frozenset({EntityType.EMAIL, EntityType.CREDIT_CARD})
SSN_DIGIT_COUNTS = frozenset({4, 9})
SSN_SHAPE = re.compile(r"[\d -]+")


def is_ssn_shaped(value: str) -> bool:
    return (
        SSN_SHAPE.fullmatch(value) is not None
        and len(digits_of(value)) in SSN_DIGIT_COUNTS
    )


def passes_format_check(text: str, span: Span) -> bool:
    value = text[span.start : span.end]
    if span.label is EntityType.CREDIT_CARD:
        return is_valid_card_number(value)
    if span.label is EntityType.PHONE:
        return has_phone_digit_count(value)
    if span.label is EntityType.SSN:
        return is_ssn_shaped(value)
    if span.label is EntityType.EMAIL:
        return EMAIL.fullmatch(value) is not None
    return True


def may_take_over(text: str, validator_span: Span, model_span: Span) -> bool:
    if validator_span.label in OVERRIDING_VALIDATOR_TYPES:
        return True
    if model_span.label is validator_span.label:
        return True
    return not passes_format_check(text, model_span)


def is_accepted(text: str, validator_span: Span, model_spans: Sequence[Span]) -> bool:
    return all(
        may_take_over(text, validator_span, model_span)
        for model_span in model_spans
        if model_span.overlaps(validator_span)
    )


def stripped(text: str, span: Span, start: int, end: int) -> Span | None:
    value = text[start:end]
    start += len(value) - len(value.lstrip())
    end -= len(value) - len(value.rstrip())
    return replace(span, start=start, end=end) if end > start else None


def pieces_outside(
    text: str, model_span: Span, validator_spans: Sequence[Span]
) -> list[Span]:
    pieces: list[Span | None] = []
    cursor = model_span.start
    for validator_span in validator_spans:
        if not model_span.overlaps(validator_span):
            continue
        if validator_span.start > cursor:
            pieces.append(stripped(text, model_span, cursor, validator_span.start))
        cursor = max(cursor, validator_span.end)
    if cursor < model_span.end:
        pieces.append(stripped(text, model_span, cursor, model_span.end))
    return [piece for piece in pieces if piece is not None]


def combine_spans(
    text: str, model_spans: Sequence[Span], validator_spans: Sequence[Span]
) -> list[Span]:
    accepted = sorted(
        (span for span in validator_spans if is_accepted(text, span, model_spans)),
        key=lambda span: span.start,
    )
    pieces = [
        piece
        for model_span in model_spans
        for piece in pieces_outside(text, model_span, accepted)
    ]
    return sorted(pieces + accepted, key=lambda span: span.start)
