from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from edgeveil.detector.alignment import Offset, decode_label_ids, token_label_ids
from edgeveil.spans import EntityType, GoldSpan, Span


class ErrorSide(StrEnum):
    MISS = "misses"
    FALSE_POSITIVE = "false_positives"


class ErrorKind(StrEnum):
    MISSED = "missed"
    SPURIOUS = "spurious"
    BOUNDARY = "boundary"
    WRONG_TYPE = "wrong_type"


KINDS_BY_SIDE = {
    ErrorSide.MISS: (ErrorKind.MISSED, ErrorKind.BOUNDARY, ErrorKind.WRONG_TYPE),
    ErrorSide.FALSE_POSITIVE: (
        ErrorKind.SPURIOUS,
        ErrorKind.BOUNDARY,
        ErrorKind.WRONG_TYPE,
    ),
}
NO_OVERLAP_KIND = {
    ErrorSide.MISS: ErrorKind.MISSED,
    ErrorSide.FALSE_POSITIVE: ErrorKind.SPURIOUS,
}

ErrorCounts = dict[str, dict[str, dict[str, int]]]


@dataclass(frozen=True, slots=True)
class DetectionError:
    snippet_index: int
    side: ErrorSide
    kind: ErrorKind
    span: GoldSpan
    counterpart: GoldSpan | None


def token_aligned(
    spans: Sequence[GoldSpan | Span], offsets: Sequence[Offset]
) -> list[GoldSpan]:
    gold_like = [GoldSpan(span.start, span.end, span.label) for span in spans]
    return decode_label_ids(token_label_ids(gold_like, offsets), offsets)


def overlaps(left: GoldSpan, right: GoldSpan) -> bool:
    return left.start < right.end and right.start < left.end


def classify(
    snippet_index: int, side: ErrorSide, span: GoldSpan, others: Sequence[GoldSpan]
) -> DetectionError | None:
    if span in others:
        return None
    touching = [other for other in others if overlaps(span, other)]
    same_type = [other for other in touching if other.label is span.label]
    if same_type:
        return DetectionError(
            snippet_index, side, ErrorKind.BOUNDARY, span, same_type[0]
        )
    if touching:
        return DetectionError(
            snippet_index, side, ErrorKind.WRONG_TYPE, span, touching[0]
        )
    return DetectionError(snippet_index, side, NO_OVERLAP_KIND[side], span, None)


def snippet_errors(
    snippet_index: int, gold: Sequence[GoldSpan], predicted: Sequence[GoldSpan]
) -> list[DetectionError]:
    found = [classify(snippet_index, ErrorSide.MISS, span, predicted) for span in gold]
    found += [
        classify(snippet_index, ErrorSide.FALSE_POSITIVE, span, gold)
        for span in predicted
    ]
    return [error for error in found if error is not None]


def detection_errors(
    gold: Sequence[Sequence[GoldSpan]],
    predicted: Sequence[Sequence[GoldSpan]],
) -> list[DetectionError]:
    return [
        error
        for index, (gold_spans, predicted_spans) in enumerate(
            zip(gold, predicted, strict=True)
        )
        for error in snippet_errors(index, gold_spans, predicted_spans)
    ]


def error_counts(errors: Sequence[DetectionError]) -> ErrorCounts:
    counts: ErrorCounts = {
        side.value: {
            entity.value: {kind.value: 0 for kind in kinds} for entity in EntityType
        }
        for side, kinds in KINDS_BY_SIDE.items()
    }
    for error in errors:
        counts[error.side.value][error.span.label.value][error.kind.value] += 1
    return counts
