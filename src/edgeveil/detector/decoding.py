from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from edgeveil.detector.alignment import Offset, decode_label_ids, is_special_token
from edgeveil.detector.labels import OUTSIDE_ID, begin_id, inside_id
from edgeveil.detector.windows import Window
from edgeveil.spans import EntityType, GoldSpan, Span, SpanSource, uncovered_ranges

Thresholds = Mapping[EntityType, float]
DEFAULT_THRESHOLD = 0.5

ENTITY_ORDER = tuple(EntityType)
BEGIN_IDS = np.array([begin_id(entity) for entity in ENTITY_ORDER])
INSIDE_IDS = np.array([inside_id(entity) for entity in ENTITY_ORDER])


def uniform_thresholds(value: float = DEFAULT_THRESHOLD) -> dict[EntityType, float]:
    return dict.fromkeys(ENTITY_ORDER, value)


def softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def entity_masses(probabilities: NDArray[np.floating]) -> NDArray[np.float64]:
    return probabilities[:, BEGIN_IDS] + probabilities[:, INSIDE_IDS]


def candidate_label_ids(probabilities: NDArray[np.floating]) -> list[int]:
    masses = entity_masses(probabilities)
    rows = np.arange(len(probabilities))
    best = masses.argmax(axis=1)
    is_entity = masses[rows, best] > probabilities[rows, OUTSIDE_ID]
    is_inside = (
        probabilities[rows, INSIDE_IDS[best]] > probabilities[rows, BEGIN_IDS[best]]
    )
    entity_ids = np.where(is_inside, INSIDE_IDS[best], BEGIN_IDS[best])
    return np.where(is_entity, entity_ids, OUTSIDE_ID).tolist()


def tokens_inside(span: GoldSpan, offsets: Sequence[Offset]) -> list[int]:
    return [
        index
        for index, offset in enumerate(offsets)
        if not is_special_token(offset)
        and span.start <= offset[0]
        and offset[1] <= span.end
    ]


def touches_same_type(left: GoldSpan | Span, right: GoldSpan | Span) -> bool:
    return left.label is right.label and left.end == right.start


def join_touching(spans: Sequence[GoldSpan]) -> list[GoldSpan]:
    joined: list[GoldSpan] = []
    for span in sorted(spans, key=lambda item: item.start):
        if joined and touches_same_type(joined[-1], span):
            joined[-1] = GoldSpan(joined[-1].start, span.end, span.label)
        else:
            joined.append(span)
    return joined


def window_spans(
    probabilities: NDArray[np.floating],
    offsets: Sequence[Offset],
    thresholds: Thresholds,
) -> list[Span]:
    masses = entity_masses(probabilities)
    spans: list[Span] = []
    decoded_spans = decode_label_ids(candidate_label_ids(probabilities), offsets)
    for decoded in join_touching(decoded_spans):
        column = ENTITY_ORDER.index(decoded.label)
        score = min(float(masses[tokens_inside(decoded, offsets), column].mean()), 1.0)
        if score >= thresholds[decoded.label]:
            spans.append(
                Span(decoded.start, decoded.end, decoded.label, score, SpanSource.MODEL)
            )
    return spans


def join_same_type(spans: Sequence[Span]) -> list[Span]:
    joined: list[Span] = []
    for incoming in sorted(spans, key=lambda span: (span.label.value, span.start)):
        previous = joined[-1] if joined else None
        if (
            previous is not None
            and previous.label is incoming.label
            and (previous.overlaps(incoming) or touches_same_type(previous, incoming))
        ):
            joined[-1] = replace(
                previous,
                end=max(previous.end, incoming.end),
                score=max(previous.score, incoming.score),
            )
        else:
            joined.append(incoming)
    return joined


def snapped_to_tokens(
    span: Span, start: int, end: int, offsets: Sequence[Offset]
) -> Span | None:
    inside = [
        offset
        for offset in offsets
        if not is_special_token(offset) and start <= offset[0] and offset[1] <= end
    ]
    if not inside:
        return None
    return replace(span, start=inside[0][0], end=inside[-1][1])


def merge_window_spans(spans: Sequence[Span], offsets: Sequence[Offset]) -> list[Span]:
    kept: list[Span] = []
    for span in sorted(
        join_same_type(spans), key=lambda item: (-item.score, item.start)
    ):
        for start, end in uncovered_ranges(span.start, span.end, kept):
            piece = (
                replace(span, start=start, end=end)
                if (start, end) == (span.start, span.end)
                else snapped_to_tokens(span, start, end, offsets)
            )
            if piece is not None:
                kept.append(piece)
    return sorted(kept, key=lambda span: span.start)


def logits_to_spans(
    window_logits: Sequence[NDArray[np.floating]],
    windows: Sequence[Window],
    thresholds: Thresholds,
) -> list[Span]:
    spans = [
        span
        for logits, window in zip(window_logits, windows, strict=True)
        for span in window_spans(softmax(logits), window.offsets, thresholds)
    ]
    offsets = sorted({offset for window in windows for offset in window.offsets})
    return merge_window_spans(spans, offsets)
