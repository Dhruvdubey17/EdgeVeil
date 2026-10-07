from collections.abc import Sequence
from dataclasses import dataclass

from edgeveil.detector.labels import (
    IGNORED_LABEL_ID,
    OUTSIDE_ID,
    begin_id,
    entity_of,
    inside_id,
    is_begin,
)
from edgeveil.spans import EntityType, GoldSpan

Offset = tuple[int, int]
SPECIAL_TOKEN_OFFSET: Offset = (0, 0)


def is_special_token(offset: Offset) -> bool:
    return offset[0] == offset[1]


def token_label_ids(spans: Sequence[GoldSpan], offsets: Sequence[Offset]) -> list[int]:
    ordered = sorted(spans, key=lambda span: span.start)
    labels: list[int] = []
    span_index = 0
    begun_index = -1
    for offset in offsets:
        if is_special_token(offset):
            labels.append(IGNORED_LABEL_ID)
            continue
        token_start, token_end = offset
        while span_index < len(ordered) and ordered[span_index].end <= token_start:
            span_index += 1
        if span_index == len(ordered) or ordered[span_index].start >= token_end:
            labels.append(OUTSIDE_ID)
            continue
        entity = ordered[span_index].label
        labels.append(
            inside_id(entity) if begun_index == span_index else begin_id(entity)
        )
        begun_index = span_index
    return labels


@dataclass(slots=True)
class OpenSpan:
    start: int
    end: int
    label: EntityType

    def closed(self) -> GoldSpan:
        return GoldSpan(self.start, self.end, self.label)


def continues(current: OpenSpan, label_id: int) -> bool:
    return current.label is entity_of(label_id) and not is_begin(label_id)


def decode_label_ids(
    label_ids: Sequence[int], offsets: Sequence[Offset]
) -> list[GoldSpan]:
    spans: list[GoldSpan] = []
    current: OpenSpan | None = None
    for label_id, offset in zip(label_ids, offsets, strict=True):
        if label_id == IGNORED_LABEL_ID or is_special_token(offset):
            continue
        if current is not None and continues(current, label_id):
            current.end = offset[1]
            continue
        entity = entity_of(label_id)
        if current is not None:
            spans.append(current.closed())
        current = None if entity is None else OpenSpan(offset[0], offset[1], entity)
    if current is not None:
        spans.append(current.closed())
    return spans
