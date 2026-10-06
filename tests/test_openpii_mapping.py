from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from edgeveil.dataset.openpii import (
    DIRECT_LABELS,
    IGNORED_LABELS,
    OverlappingEntitiesError,
    RawEntity,
    UnknownLabelError,
    map_entities,
)
from edgeveil.spans import EntityType, GoldSpan

Part = str | tuple[str, str]


def compose(*parts: Part) -> tuple[str, list[RawEntity]]:
    text = ""
    entities: list[RawEntity] = []
    for part in parts:
        if isinstance(part, str):
            text += part
            continue
        label, value = part
        entities.append(
            RawEntity(
                start=len(text), end=len(text) + len(value), label=label, value=value
            )
        )
        text += value
    return text, entities


def mapped(*parts: Part) -> list[tuple[str, EntityType]]:
    text, entities = compose(*parts)
    return [
        (text[span.start : span.end], span.label)
        for span in map_entities(text, entities)
    ]


def test_given_name_and_surname_across_space_become_one_person() -> None:
    assert mapped("Hi ", ("GIVENNAME", "Ada"), " ", ("SURNAME", "Quill"), "!") == [
        ("Ada Quill", EntityType.PERSON)
    ]


def test_person_parts_merge_across_any_whitespace() -> None:
    assert mapped(("GIVENNAME", "Ada"), " \t ", ("SURNAME", "Quill")) == [
        ("Ada \t Quill", EntityType.PERSON)
    ]


def test_person_parts_never_merge_across_comma() -> None:
    assert mapped(("SURNAME", "Quill"), ", ", ("GIVENNAME", "Ada")) == [
        ("Quill", EntityType.PERSON),
        ("Ada", EntityType.PERSON),
    ]


@pytest.mark.parametrize("label", ["GIVENNAME", "SURNAME"])
def test_lone_name_part_is_person(label: str) -> None:
    assert mapped("Ask ", (label, "Quill"), ".") == [("Quill", EntityType.PERSON)]


def test_title_stays_outside_person() -> None:
    assert mapped(("TITLE", "Dr."), " ", ("SURNAME", "Quill")) == [
        ("Quill", EntityType.PERSON)
    ]


def test_full_address_with_region_code_becomes_one_span() -> None:
    assert mapped(
        "Send it to ",
        ("BUILDINGNUM", "12"),
        " ",
        ("STREET", "Imaginary Lane"),
        ", ",
        ("CITY", "Nowhere City"),
        ", ZZ ",
        ("ZIPCODE", "00000"),
        " today.",
    ) == [("12 Imaginary Lane, Nowhere City, ZZ 00000", EntityType.ADDRESS)]


@pytest.mark.parametrize("separator", [" ", ", ", ". ", " #", "\n", ",\n"])
def test_address_parts_merge_across_separators(separator: str) -> None:
    assert mapped(("STREET", "Imaginary Lane"), separator, ("BUILDINGNUM", "4")) == [
        (f"Imaginary Lane{separator}4", EntityType.ADDRESS)
    ]


@pytest.mark.parametrize("gap", [", zz ", ", ZZZ ", "ZZ ", " near ", " - "])
def test_address_parts_do_not_merge_across_other_text(gap: str) -> None:
    assert mapped(("STREET", "Imaginary Lane"), gap, ("ZIPCODE", "00000")) == [
        ("Imaginary Lane", EntityType.ADDRESS),
        ("00000", EntityType.ADDRESS),
    ]


def test_city_alone_is_not_an_address() -> None:
    assert mapped("I moved to ", ("CITY", "Nowhere City"), ".") == []


def test_cities_without_an_anchor_part_are_not_an_address() -> None:
    assert mapped(("CITY", "Nowhere City"), ", ", ("CITY", "Elsewhere")) == []


@pytest.mark.parametrize("anchor", ["STREET", "BUILDINGNUM", "ZIPCODE"])
def test_city_joins_an_address_with_an_anchor_part(anchor: str) -> None:
    assert mapped(("CITY", "Nowhere City"), " ", (anchor, "00000")) == [
        ("Nowhere City 00000", EntityType.ADDRESS)
    ]


def test_person_and_address_parts_never_merge() -> None:
    assert mapped(("SURNAME", "Quill"), " ", ("STREET", "Imaginary Lane")) == [
        ("Quill", EntityType.PERSON),
        ("Imaginary Lane", EntityType.ADDRESS),
    ]


@pytest.mark.parametrize(("label", "expected"), sorted(DIRECT_LABELS.items()))
def test_direct_labels_map_one_to_one(label: str, expected: EntityType) -> None:
    assert mapped("x ", (label, "value-1"), " y") == [("value-1", expected)]


def test_government_id_labels_map_to_gov_id() -> None:
    gov_id_labels = {
        label for label, target in DIRECT_LABELS.items() if target is EntityType.GOV_ID
    }
    assert gov_id_labels == {"IDCARDNUM", "PASSPORTNUM", "DRIVERLICENSENUM", "TAXNUM"}


@pytest.mark.parametrize("label", sorted(IGNORED_LABELS))
def test_ignored_labels_produce_no_span(label: str) -> None:
    assert mapped("x ", (label, "value"), " y") == []


def test_ignored_labels_are_exactly_the_agreed_set() -> None:
    assert {"DATE", "AGE", "GENDER", "SEX", "TITLE"} == IGNORED_LABELS


def test_accepts_entities_in_any_order() -> None:
    text, entities = compose(("GIVENNAME", "Ada"), " ", ("SURNAME", "Quill"))
    assert map_entities(text, list(reversed(entities))) == [
        GoldSpan(0, 9, EntityType.PERSON)
    ]


def test_unknown_label_raises() -> None:
    text, entities = compose(("FAVOURITECOLOUR", "teal"))
    with pytest.raises(UnknownLabelError):
        map_entities(text, entities)


def test_overlapping_entities_raise() -> None:
    entities = [
        RawEntity(start=0, end=5, label="EMAIL", value="abcde"),
        RawEntity(start=3, end=8, label="TELEPHONENUM", value="defgh"),
    ]
    with pytest.raises(OverlappingEntitiesError):
        map_entities("abcdefgh", entities)


def test_raw_entity_repr_hides_value() -> None:
    entity = RawEntity(start=0, end=16, label="EMAIL", value="ada@example.com")
    assert "example.com" not in repr(entity)


known_labels = st.sampled_from(
    sorted(
        {
            *DIRECT_LABELS,
            *IGNORED_LABELS,
            "GIVENNAME",
            "SURNAME",
            "STREET",
            "BUILDINGNUM",
            "CITY",
            "ZIPCODE",
        }
    )
)
gaps = st.text(alphabet=" ,.#\nZaz-", max_size=4)
values = st.text(alphabet="abc123", min_size=1, max_size=6)
segments = st.lists(st.tuples(gaps, known_labels, values), max_size=12)


@given(segments=segments)
def test_output_is_sorted_disjoint_and_keeps_direct_entities(
    segments: list[tuple[str, str, str]],
) -> None:
    text, entities = compose(
        *[part for gap, label, value in segments for part in (gap, (label, value))]
    )
    spans = map_entities(text, entities)
    assert all(0 <= span.start < span.end <= len(text) for span in spans)
    assert all(left.end <= right.start for left, right in pairwise(spans))
    direct = [
        GoldSpan(entity.start, entity.end, DIRECT_LABELS[entity.label])
        for entity in entities
        if entity.label in DIRECT_LABELS
    ]
    assert set(direct) <= set(spans)
