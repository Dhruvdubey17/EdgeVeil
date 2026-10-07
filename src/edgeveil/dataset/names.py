import random
import string
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from edgeveil.dataset.openpii import NAME_PARTICLES, PARENTAGE_MARKERS, DatasetError
from edgeveil.spans import GoldSpan, LabeledText

NAME_CONNECTOR_PAIR_FRACTION = 0.03
INITIAL_CONNECTOR_SHARE = 0.5
INITIAL_PERIOD_OPTIONS = ("", ".")

NameJoint = tuple[int, int]


class JointOutsidePersonError(DatasetError):
    pass


@dataclass(frozen=True, slots=True)
class NameConnectorResult:
    rows: list[LabeledText]
    eligible_pairs: int
    connectors_inserted: int


def name_connector(rng: random.Random) -> str:
    if rng.random() < INITIAL_CONNECTOR_SHARE:
        return rng.choice(string.ascii_uppercase) + rng.choice(INITIAL_PERIOD_OPTIONS)
    return rng.choice(NAME_PARTICLES + PARENTAGE_MARKERS)


def contains_joint(span: GoldSpan, joint: NameJoint) -> bool:
    start, end = joint
    return span.start <= start and end <= span.end


def moved_span(span: GoldSpan, joint: NameJoint, growth: int) -> GoldSpan:
    if contains_joint(span, joint):
        return GoldSpan(span.start, span.end + growth, span.label)
    if span.start >= joint[1]:
        return GoldSpan(span.start + growth, span.end + growth, span.label)
    return span


def insert_name_connector(
    row: LabeledText, joint: NameJoint, connector: str
) -> LabeledText:
    if not any(contains_joint(span, joint) for span in row.spans):
        raise JointOutsidePersonError(f"name joint {joint} is not inside a span")
    start, end = joint
    gap = row.text[start:end]
    replacement = gap + connector + gap
    growth = len(replacement) - len(gap)
    text = row.text[:start] + replacement + row.text[end:]
    spans = tuple(moved_span(span, joint, growth) for span in row.spans)
    return LabeledText(text=text, spans=spans)


def add_name_connectors(
    rows: Sequence[LabeledText],
    joints: Sequence[Sequence[NameJoint]],
    rng: random.Random,
) -> NameConnectorResult:
    candidates = [
        (index, joint)
        for index, row_joints in enumerate(joints)
        for joint in row_joints
    ]
    chosen = rng.sample(
        candidates, round(len(candidates) * NAME_CONNECTOR_PAIR_FRACTION)
    )
    chosen_by_row: defaultdict[int, list[NameJoint]] = defaultdict(list)
    for index, joint in chosen:
        chosen_by_row[index].append(joint)
    updated = list(rows)
    for index, row_joints in chosen_by_row.items():
        for joint in sorted(row_joints, reverse=True):
            updated[index] = insert_name_connector(
                updated[index], joint, name_connector(rng)
            )
    return NameConnectorResult(
        rows=updated, eligible_pairs=len(candidates), connectors_inserted=len(chosen)
    )
