import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise

from edgeveil.spans import EntityType, GoldSpan

OPENPII_REPOSITORY = "ai4privacy/pii-masking-openpii-1.5m"
OPENPII_REVISION = "a785eb528e28be2693c3718a27e066970de5dadb"
OPENPII_SPLITS = ("train", "validation")
ENGLISH_LANGUAGE_CODE = "en"

DIRECT_LABELS: dict[str, EntityType] = {
    "EMAIL": EntityType.EMAIL,
    "TELEPHONENUM": EntityType.PHONE,
    "SOCIALNUM": EntityType.SSN,
    "CREDITCARDNUMBER": EntityType.CREDIT_CARD,
    "IDCARDNUM": EntityType.GOV_ID,
    "PASSPORTNUM": EntityType.GOV_ID,
    "DRIVERLICENSENUM": EntityType.GOV_ID,
    "TAXNUM": EntityType.GOV_ID,
}
PERSON_PARTS = frozenset({"GIVENNAME", "SURNAME"})
ADDRESS_PARTS = frozenset({"STREET", "BUILDINGNUM", "CITY", "ZIPCODE"})
ADDRESS_ANCHORS = frozenset({"STREET", "BUILDINGNUM", "ZIPCODE"})
IGNORED_LABELS = frozenset({"DATE", "AGE", "GENDER", "SEX", "TITLE"})

MAX_ADDRESS_GAP_LENGTH = 16
ADDRESS_UNIT_WORDS = (
    "apartment",
    "suite",
    "floor",
    "level",
    "block",
    "flat",
    "unit",
    "room",
    "bldg",
    "apt",
    "ste",
    "blk",
    "fl",
    "rm",
    "no",
)
NAME_PARTICLES = ("binti", "bin", "van", "von", "del", "de", "da", "la", "le", "al")
PARENTAGE_MARKERS = ("s/o", "d/o")


def alternation(words: Iterable[str]) -> str:
    return "|".join(re.escape(word) for word in words)


ADDRESS_SEPARATOR = r"[\s,.#\-\u2013\u2014]"
ADDRESS_WORD = rf"(?:(?i:{alternation(ADDRESS_UNIT_WORDS)})|[A-Z]{{2}})"
ADDRESS_GAP = re.compile(
    rf"{ADDRESS_SEPARATOR}*"
    rf"|{ADDRESS_SEPARATOR}+{ADDRESS_WORD}(?:{ADDRESS_SEPARATOR}+{ADDRESS_WORD})*"
    rf"{ADDRESS_SEPARATOR}*"
)
MIDDLE_INITIAL = r"[A-Z]\.?"
NAME_CONNECTOR = (
    rf"(?:{MIDDLE_INITIAL}|{alternation(NAME_PARTICLES + PARENTAGE_MARKERS)})"
)
PERSON_GAP = re.compile(rf"\s*|\s+(?:{NAME_CONNECTOR}\s+)+")


class DatasetError(Exception):
    pass


class UnknownLabelError(DatasetError):
    pass


class OverlappingEntitiesError(DatasetError):
    pass


class MalformedEntityError(DatasetError):
    pass


class PartKind(StrEnum):
    PERSON = "person"
    ADDRESS = "address"


@dataclass(frozen=True, slots=True)
class RawEntity:
    start: int
    end: int
    label: str
    value: str = field(repr=False)


def map_entities(text: str, entities: Iterable[RawEntity]) -> list[GoldSpan]:
    ordered = sorted(entities, key=lambda entity: (entity.start, entity.end))
    reject_unknown_labels(ordered)
    reject_overlaps(ordered)
    spans: list[GoldSpan] = []
    for group in group_mergeable_parts(text, trimmed_entities(text, ordered)):
        span = span_for_group(group)
        if span is not None:
            spans.append(span)
    return spans


def unknown_labels(entities: Iterable[RawEntity]) -> list[str]:
    known = DIRECT_LABELS.keys() | PERSON_PARTS | ADDRESS_PARTS | IGNORED_LABELS
    return [entity.label for entity in entities if entity.label not in known]


def reject_unknown_labels(entities: Sequence[RawEntity]) -> None:
    unknown = sorted(set(unknown_labels(entities)))
    if unknown:
        raise UnknownLabelError(f"unmapped OpenPII labels: {', '.join(unknown)}")


def trim_whitespace(text: str, entity: RawEntity) -> RawEntity | None:
    value = text[entity.start : entity.end]
    without_leading = value.lstrip()
    start = entity.end - len(without_leading)
    end = start + len(without_leading.rstrip())
    if end == start:
        return None
    return RawEntity(start=start, end=end, label=entity.label, value=text[start:end])


def trimmed_entities(text: str, entities: Iterable[RawEntity]) -> list[RawEntity]:
    trimmed: list[RawEntity] = []
    for entity in entities:
        kept = trim_whitespace(text, entity)
        if kept is not None:
            trimmed.append(kept)
    return trimmed


def reject_overlaps(entities: Sequence[RawEntity]) -> None:
    for left, right in pairwise(entities):
        if right.start < left.end:
            raise OverlappingEntitiesError(
                f"{left.label} [{left.start}, {left.end}) overlaps "
                f"{right.label} [{right.start}, {right.end})"
            )


def part_kind(label: str) -> PartKind | None:
    if label in PERSON_PARTS:
        return PartKind.PERSON
    if label in ADDRESS_PARTS:
        return PartKind.ADDRESS
    return None


def can_merge(text: str, left: RawEntity, right: RawEntity) -> bool:
    kind = part_kind(left.label)
    if kind is None or kind is not part_kind(right.label):
        return False
    gap = text[left.end : right.start]
    if kind is PartKind.PERSON:
        return is_person_gap(gap)
    return is_address_gap(gap)


def is_person_gap(gap: str) -> bool:
    return PERSON_GAP.fullmatch(gap) is not None


def is_address_gap(gap: str) -> bool:
    if len(gap) > MAX_ADDRESS_GAP_LENGTH:
        return False
    return ADDRESS_GAP.fullmatch(gap) is not None


def group_mergeable_parts(
    text: str, entities: Sequence[RawEntity]
) -> list[list[RawEntity]]:
    groups: list[list[RawEntity]] = []
    for entity in entities:
        if groups and can_merge(text, groups[-1][-1], entity):
            groups[-1].append(entity)
        else:
            groups.append([entity])
    return groups


def span_for_group(group: Sequence[RawEntity]) -> GoldSpan | None:
    first, last = group[0], group[-1]
    label = label_for_group(group)
    if label is None:
        return None
    return GoldSpan(start=first.start, end=last.end, label=label)


def label_for_group(group: Sequence[RawEntity]) -> EntityType | None:
    kind = part_kind(group[0].label)
    if kind is PartKind.PERSON:
        return EntityType.PERSON
    if kind is PartKind.ADDRESS:
        has_anchor = any(entity.label in ADDRESS_ANCHORS for entity in group)
        return EntityType.ADDRESS if has_anchor else None
    return DIRECT_LABELS.get(group[0].label)


def parse_privacy_mask(records: Iterable[Mapping[str, object]]) -> list[RawEntity]:
    return [parse_privacy_mask_record(record) for record in records]


def parse_privacy_mask_record(record: Mapping[str, object]) -> RawEntity:
    start, end = record.get("start"), record.get("end")
    label, value = record.get("label"), record.get("value")
    if not isinstance(start, int) or not isinstance(end, int):
        raise MalformedEntityError("privacy_mask offsets are not integers")
    if not isinstance(label, str) or not isinstance(value, str):
        raise MalformedEntityError(
            f"privacy_mask entry [{start}, {end}) lacks a string label or value"
        )
    return RawEntity(start=start, end=end, label=label, value=value)


def mismatched_labels(text: str, entities: Iterable[RawEntity]) -> list[str]:
    return [
        entity.label
        for entity in entities
        if text[entity.start : entity.end] != entity.value
    ]


@dataclass(frozen=True, slots=True)
class SplitDeduplication:
    kept_train: list[int]
    kept_validation: list[int]
    train_dropped_as_in_validation: int
    train_dropped_as_repeat: int
    validation_dropped_as_repeat: int


def first_occurrences(texts: Sequence[str], excluded: set[str]) -> list[int]:
    seen = set(excluded)
    kept: list[int] = []
    for index, text in enumerate(texts):
        if text not in seen:
            kept.append(index)
            seen.add(text)
    return kept


def deduplicate_splits(
    train: Sequence[str], validation: Sequence[str]
) -> SplitDeduplication:
    kept_validation = first_occurrences(validation, excluded=set())
    validation_texts = set(validation)
    train_outside_validation = [text for text in train if text not in validation_texts]
    kept_train = first_occurrences(train, excluded=validation_texts)
    return SplitDeduplication(
        kept_train=kept_train,
        kept_validation=kept_validation,
        train_dropped_as_in_validation=len(train) - len(train_outside_validation),
        train_dropped_as_repeat=len(train_outside_validation) - len(kept_train),
        validation_dropped_as_repeat=len(validation) - len(kept_validation),
    )


def is_name_joint(text: str, left: RawEntity, right: RawEntity) -> bool:
    if left.label != "GIVENNAME" or right.label != "SURNAME":
        return False
    gap = text[left.end : right.start]
    return gap.isspace()


def name_joints(text: str, entities: Iterable[RawEntity]) -> list[tuple[int, int]]:
    ordered = sorted(entities, key=lambda entity: (entity.start, entity.end))
    trimmed = trimmed_entities(text, ordered)
    return [
        (left.end, right.start)
        for left, right in pairwise(trimmed)
        if is_name_joint(text, left, right)
    ]
