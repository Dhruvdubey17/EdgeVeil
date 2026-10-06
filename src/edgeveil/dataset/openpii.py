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

PERSON_GAP = re.compile(r"\s*")
ADDRESS_GAP = re.compile(r"[\s,.#]*(?:(?<=[\s,.#])[A-Z]{2}[\s,.#]+)?")


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
    for group in group_mergeable_parts(text, ordered):
        span = span_for_group(group)
        if span is not None:
            spans.append(span)
    return spans


def reject_unknown_labels(entities: Sequence[RawEntity]) -> None:
    known = DIRECT_LABELS.keys() | PERSON_PARTS | ADDRESS_PARTS | IGNORED_LABELS
    unknown = sorted({entity.label for entity in entities} - known)
    if unknown:
        raise UnknownLabelError(f"unmapped OpenPII labels: {', '.join(unknown)}")


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
    pattern = PERSON_GAP if kind is PartKind.PERSON else ADDRESS_GAP
    return pattern.fullmatch(gap) is not None


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
