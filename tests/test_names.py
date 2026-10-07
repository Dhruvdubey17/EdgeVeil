import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from edgeveil.dataset.names import (
    NAME_CONNECTOR_PAIR_FRACTION,
    add_name_connectors,
    insert_name_connector,
    name_connector,
)
from edgeveil.dataset.openpii import is_person_gap
from edgeveil.spans import EntityType, GoldSpan, LabeledText

TEXT = "Ask Ada Quill or mail bo@example.com."
ROW = LabeledText(
    text=TEXT,
    spans=(GoldSpan(4, 13, EntityType.PERSON), GoldSpan(22, 36, EntityType.EMAIL)),
)
JOINT = (7, 8)


def span_texts(row: LabeledText) -> list[str]:
    return [row.text[span.start : span.end] for span in row.spans]


def test_inserting_a_connector_widens_the_person_and_shifts_later_spans() -> None:
    result = insert_name_connector(ROW, JOINT, "B.")
    assert result.text == "Ask Ada B. Quill or mail bo@example.com."
    assert span_texts(result) == ["Ada B. Quill", "bo@example.com"]


@pytest.mark.parametrize("seed", range(50))
def test_connectors_are_initials_or_listed_particles(seed: int) -> None:
    connector = name_connector(random.Random(seed))
    assert is_person_gap(f" {connector} ")


@given(seed=st.integers(min_value=0, max_value=10_000))
def test_connectors_keep_person_gaps_mergeable(seed: int) -> None:
    result = insert_name_connector(ROW, JOINT, name_connector(random.Random(seed)))
    person = result.spans[0]
    assert result.text[person.start : person.end].startswith("Ada ")
    assert result.text[person.start : person.end].endswith(" Quill")
    assert span_texts(result)[1] == "bo@example.com"


def test_add_name_connectors_hits_the_target_fraction() -> None:
    rows = [ROW] * 1000
    joints = [[JOINT]] * 1000
    result = add_name_connectors(rows, joints, random.Random(3))
    assert result.connectors_inserted == round(1000 * NAME_CONNECTOR_PAIR_FRACTION)
    assert result.eligible_pairs == 1000
    changed = sum(row.text != TEXT for row in result.rows)
    assert changed == result.connectors_inserted


def test_several_joints_in_one_row_all_stay_aligned() -> None:
    text = "Ada Quill and Bo Lark"
    row = LabeledText(
        text=text,
        spans=(GoldSpan(0, 9, EntityType.PERSON), GoldSpan(14, 21, EntityType.PERSON)),
    )
    result = insert_name_connector(
        insert_name_connector(row, (16, 17), "van"), (3, 4), "C"
    )
    assert span_texts(result) == ["Ada C Quill", "Bo van Lark"]
