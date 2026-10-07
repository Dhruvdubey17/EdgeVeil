from edgeveil.dataset.gaps import LONG_GAP_LABEL, adjacent_parts, displayed_gap
from edgeveil.dataset.openpii import MAX_ADDRESS_GAP_LENGTH, PartKind, RawEntity

TEXT = "Ada and Quill live at 12 Imaginary Lane in Nowhere City."
ENTITIES = [
    RawEntity(0, 3, "GIVENNAME", "Ada"),
    RawEntity(8, 13, "SURNAME", "Quill"),
    RawEntity(22, 24, "BUILDINGNUM", "12"),
    RawEntity(25, 39, "STREET", "Imaginary Lane"),
    RawEntity(43, 55, "CITY", "Nowhere City"),
]


def test_adjacent_parts_report_merged_and_split_pairs() -> None:
    pairs = adjacent_parts(TEXT, ENTITIES)
    assert [(pair.kind, pair.merged) for pair in pairs] == [
        (PartKind.PERSON, False),
        (PartKind.ADDRESS, True),
        (PartKind.ADDRESS, False),
    ]
    assert [pair.gap for pair in pairs if not pair.merged] == [" and ", " in "]


def test_long_gaps_are_bucketed_at_the_address_gap_cap() -> None:
    assert displayed_gap("," * MAX_ADDRESS_GAP_LENGTH) == repr(
        "," * MAX_ADDRESS_GAP_LENGTH
    )
    assert displayed_gap("," * (MAX_ADDRESS_GAP_LENGTH + 1)) == LONG_GAP_LABEL
