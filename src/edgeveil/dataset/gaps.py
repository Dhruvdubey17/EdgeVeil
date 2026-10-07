from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from edgeveil.dataset.openpii import (
    MAX_ADDRESS_GAP_LENGTH,
    PartKind,
    RawEntity,
    can_merge,
    mismatched_labels,
    parse_privacy_mask,
    part_kind,
    trimmed_entities,
    unknown_labels,
)
from edgeveil.dataset.prepare import load_raw_split

DEFAULT_TOP_GAPS = 20
LONG_GAP_LABEL = f"(longer than {MAX_ADDRESS_GAP_LENGTH} characters)"


@dataclass(frozen=True, slots=True)
class AdjacentParts:
    kind: PartKind
    left_label: str
    right_label: str
    gap: str = field(repr=False)
    merged: bool


@dataclass(frozen=True, slots=True)
class GapReport:
    adjacent_pairs: dict[str, int]
    split_pairs: dict[str, int]
    top_split_gaps: dict[str, list[tuple[str, str, str, int]]]


def adjacent_parts(text: str, entities: Iterable[RawEntity]) -> list[AdjacentParts]:
    ordered = sorted(entities, key=lambda entity: (entity.start, entity.end))
    pairs: list[AdjacentParts] = []
    for left, right in pairwise(trimmed_entities(text, ordered)):
        kind = part_kind(left.label)
        if kind is None or kind is not part_kind(right.label):
            continue
        pairs.append(
            AdjacentParts(
                kind=kind,
                left_label=left.label,
                right_label=right.label,
                gap=text[left.end : right.start],
                merged=can_merge(text, left, right),
            )
        )
    return pairs


def displayed_gap(gap: str) -> str:
    if len(gap) > MAX_ADDRESS_GAP_LENGTH:
        return LONG_GAP_LABEL
    return repr(gap)


def gap_report(data_dir: Path, split: str, top: int = DEFAULT_TOP_GAPS) -> GapReport:
    adjacent: Counter[str] = Counter()
    splits: dict[str, Counter[tuple[str, str, str]]] = {
        kind.value: Counter() for kind in PartKind
    }
    for record in load_raw_split(data_dir, split):
        text = record["source_text"]
        entities = parse_privacy_mask(record["privacy_mask"])
        if mismatched_labels(text, entities) or unknown_labels(entities):
            continue
        for pair in adjacent_parts(text, entities):
            adjacent[pair.kind.value] += 1
            if not pair.merged:
                key = (pair.left_label, pair.right_label, displayed_gap(pair.gap))
                splits[pair.kind.value][key] += 1
    return GapReport(
        adjacent_pairs=dict(adjacent),
        split_pairs={kind: sum(counter.values()) for kind, counter in splits.items()},
        top_split_gaps={
            kind: [(*key, count) for key, count in counter.most_common(top)]
            for kind, counter in splits.items()
        },
    )
