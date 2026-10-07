from hypothesis import given
from hypothesis import strategies as st

from edgeveil.detector.alignment import (
    SPECIAL_TOKEN_OFFSET,
    decode_label_ids,
    token_label_ids,
)
from edgeveil.detector.labels import (
    BIO_LABELS,
    IGNORED_LABEL_ID,
    LABEL_TO_ID,
    OUTSIDE_LABEL,
)
from edgeveil.spans import EntityType, GoldSpan

Offset = tuple[int, int]


def ids(*labels: str) -> list[int]:
    return [
        IGNORED_LABEL_ID if label == "-" else LABEL_TO_ID[label] for label in labels
    ]


TEXT = "Ada Quill emailed ada@example.com"
OFFSETS: list[Offset] = [
    SPECIAL_TOKEN_OFFSET,
    (0, 3),
    (4, 7),
    (7, 9),
    (10, 17),
    (18, 21),
    (21, 22),
    (22, 33),
    SPECIAL_TOKEN_OFFSET,
]
SPANS = [GoldSpan(0, 9, EntityType.PERSON), GoldSpan(18, 33, EntityType.EMAIL)]


def test_label_map_has_outside_plus_begin_and_inside_per_type() -> None:
    assert BIO_LABELS[0] == OUTSIDE_LABEL
    assert len(BIO_LABELS) == 1 + 2 * len(EntityType)
    assert {"B-GOV_ID", "I-CREDENTIAL"} <= set(BIO_LABELS)


def test_encodes_first_token_as_begin_and_the_rest_as_inside() -> None:
    assert token_label_ids(SPANS, OFFSETS) == ids(
        "-",
        "B-PERSON",
        "I-PERSON",
        "I-PERSON",
        "O",
        "B-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "-",
    )


def test_decodes_back_to_the_same_spans() -> None:
    assert decode_label_ids(token_label_ids(SPANS, OFFSETS), OFFSETS) == SPANS


def test_adjacent_spans_of_one_type_stay_separate() -> None:
    offsets: list[Offset] = [(0, 3), (4, 9)]
    spans = [GoldSpan(0, 3, EntityType.PERSON), GoldSpan(4, 9, EntityType.PERSON)]
    labels = token_label_ids(spans, offsets)
    assert labels == ids("B-PERSON", "B-PERSON")
    assert decode_label_ids(labels, offsets) == spans


def test_span_boundary_inside_a_token_widens_to_the_token() -> None:
    offsets: list[Offset] = [(0, 5)]
    assert decode_label_ids(
        token_label_ids([GoldSpan(0, 4, EntityType.CREDENTIAL)], offsets), offsets
    ) == [GoldSpan(0, 5, EntityType.CREDENTIAL)]


def test_inside_label_without_begin_starts_a_span() -> None:
    offsets: list[Offset] = [(0, 3), (4, 7), (8, 11)]
    labels = ids("O", "I-PHONE", "I-PHONE")
    assert decode_label_ids(labels, offsets) == [GoldSpan(4, 11, EntityType.PHONE)]


def test_inside_label_of_another_type_starts_a_new_span() -> None:
    offsets: list[Offset] = [(0, 3), (4, 7)]
    labels = ids("B-PHONE", "I-EMAIL")
    assert decode_label_ids(labels, offsets) == [
        GoldSpan(0, 3, EntityType.PHONE),
        GoldSpan(4, 7, EntityType.EMAIL),
    ]


def test_special_and_ignored_tokens_never_join_spans() -> None:
    offsets: list[Offset] = [(0, 3), SPECIAL_TOKEN_OFFSET, (4, 7)]
    labels = ids("B-PHONE", "-", "I-PHONE")
    assert decode_label_ids(labels, offsets) == [GoldSpan(0, 7, EntityType.PHONE)]


@st.composite
def tokenized_text_with_spans(draw: st.DrawFn) -> tuple[list[Offset], list[GoldSpan]]:
    word_lengths = draw(
        st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=20)
    )
    offsets: list[Offset] = [SPECIAL_TOKEN_OFFSET]
    words: list[Offset] = []
    cursor = 0
    for length in word_lengths:
        cut = draw(st.integers(min_value=0, max_value=length - 1))
        if cut:
            offsets += [(cursor, cursor + cut), (cursor + cut, cursor + length)]
        else:
            offsets.append((cursor, cursor + length))
        words.append((cursor, cursor + length))
        cursor += length + 1
    offsets.append(SPECIAL_TOKEN_OFFSET)
    spans: list[GoldSpan] = []
    index = 0
    while index < len(words):
        run = draw(st.integers(min_value=0, max_value=3))
        label = draw(st.sampled_from(list(EntityType)))
        if run and index + run <= len(words):
            spans.append(GoldSpan(words[index][0], words[index + run - 1][1], label))
            index += run
        index += 1
    return offsets, spans


@given(case=tokenized_text_with_spans())
def test_round_trip_for_spans_on_token_boundaries(
    case: tuple[list[Offset], list[GoldSpan]],
) -> None:
    offsets, spans = case
    assert decode_label_ids(token_label_ids(spans, offsets), offsets) == spans
