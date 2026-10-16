from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from edgeveil.detector.alignment import Offset, decode_label_ids, is_special_token
from edgeveil.detector.labels import OUTSIDE_ID, begin_id, inside_id
from edgeveil.detector.windows import Window
from edgeveil.spans import EntityType, GoldSpan, Span, SpanSource

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


def merged_pair(kept: Span, incoming: Span) -> Span:
    if kept.label is incoming.label:
        return Span(
            kept.start,
            max(kept.end, incoming.end),
            kept.label,
            max(kept.score, incoming.score),
            SpanSource.MODEL,
        )
    return incoming if incoming.score > kept.score else kept


def merge_window_spans(spans: Sequence[Span]) -> list[Span]:
    merged: list[Span] = []
    for incoming in sorted(spans, key=lambda span: (span.start, -span.score)):
        if merged and (
            merged[-1].overlaps(incoming) or touches_same_type(merged[-1], incoming)
        ):
            merged[-1] = merged_pair(merged[-1], incoming)
        else:
            merged.append(incoming)
    return merged


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
    return merge_window_spans(spans)
