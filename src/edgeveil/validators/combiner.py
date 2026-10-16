from collections.abc import Sequence
from dataclasses import replace

from edgeveil.spans import EntityType, Span
from edgeveil.validators.finders import (
    MIN_PHONE_DIGITS,
    digits_of,
    is_valid_card_number,
)


def passes_hard_check(text: str, span: Span) -> bool:
    value = text[span.start : span.end]
    if span.label is EntityType.CREDIT_CARD:
        return is_valid_card_number(value)
    if span.label is EntityType.PHONE:
        return len(digits_of(value)) >= MIN_PHONE_DIGITS
    return True


def strictly_contains(outer: Span, inner: Span) -> bool:
    covers = outer.start <= inner.start and inner.end <= outer.end
    return covers and (outer.start, outer.end) != (inner.start, inner.end)


def model_span_wins(model_span: Span, validator_span: Span) -> bool:
    return model_span.label is not validator_span.label and strictly_contains(
        model_span, validator_span
    )


def stripped(text: str, span: Span, start: int, end: int) -> Span | None:
    value = text[start:end]
    start += len(value) - len(value.lstrip())
    end -= len(value) - len(value.rstrip())
    return replace(span, start=start, end=end) if end > start else None


def outside_part(text: str, span: Span, validator_span: Span) -> Span | None:
    if span.label is validator_span.label:
        return None
    if span.start < validator_span.start:
        return stripped(text, span, span.start, validator_span.start)
    if validator_span.end < span.end:
        return stripped(text, span, validator_span.end, span.end)
    return None


def resolved_model_span(
    text: str, model_span: Span, validator_spans: Sequence[Span]
) -> Span | None:
    current: Span | None = model_span
    for validator_span in validator_spans:
        if current is None:
            return None
        if current.overlaps(validator_span) and not model_span_wins(
            current, validator_span
        ):
            current = outside_part(text, current, validator_span)
    return current


def survives_model(validator_span: Span, model_spans: Sequence[Span]) -> bool:
    return not any(
        model_span_wins(model_span, validator_span)
        for model_span in model_spans
        if model_span.overlaps(validator_span)
    )


def combine_spans(
    text: str, model_spans: Sequence[Span], validator_spans: Sequence[Span]
) -> list[Span]:
    checked = [span for span in model_spans if passes_hard_check(text, span)]
    resolved = [resolved_model_span(text, span, validator_spans) for span in checked]
    kept_model = [span for span in resolved if span is not None]
    kept_validators = [
        span for span in validator_spans if survives_model(span, checked)
    ]
    return sorted(kept_model + kept_validators, key=lambda span: span.start)
