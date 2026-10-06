import pytest

from edgeveil.dataset.openpii import (
    MalformedEntityError,
    RawEntity,
    deduplicate_splits,
    mismatched_labels,
    parse_privacy_mask,
)

TEXT = "Mail ada@example.com or call 555-0100."
EMAIL = RawEntity(start=5, end=20, label="EMAIL", value="ada@example.com")
PHONE = RawEntity(start=29, end=37, label="TELEPHONENUM", value="555-0100")


def test_matching_offsets_report_no_mismatch() -> None:
    assert mismatched_labels(TEXT, [EMAIL, PHONE]) == []


def test_shifted_offsets_report_the_label_only() -> None:
    shifted = RawEntity(start=6, end=21, label="EMAIL", value="ada@example.com")
    assert mismatched_labels(TEXT, [shifted, PHONE]) == ["EMAIL"]


def test_out_of_range_offsets_are_a_mismatch() -> None:
    beyond = RawEntity(start=36, end=44, label="TELEPHONENUM", value="555-0100")
    assert mismatched_labels(TEXT, [beyond]) == ["TELEPHONENUM"]


def test_parses_privacy_mask_records() -> None:
    records = [
        {
            "value": "555-0100",
            "start": 29,
            "end": 37,
            "label": "TELEPHONENUM",
            "label_index": 1,
        }
    ]
    assert parse_privacy_mask(records) == [PHONE]


@pytest.mark.parametrize(
    "record",
    [
        {"value": "x", "start": "0", "end": 1, "label": "EMAIL"},
        {"value": "x", "start": 0, "end": 1},
        {"value": None, "start": 0, "end": 1, "label": "EMAIL"},
    ],
)
def test_rejects_malformed_records(record: dict[str, object]) -> None:
    with pytest.raises(MalformedEntityError):
        parse_privacy_mask([record])


def test_malformed_record_error_hides_value() -> None:
    with pytest.raises(MalformedEntityError) as raised:
        parse_privacy_mask(
            [{"value": "ada@example.com", "start": "5", "end": 20, "label": "EMAIL"}]
        )
    assert "example.com" not in str(raised.value)


def test_validation_wins_when_a_text_is_in_both_splits() -> None:
    result = deduplicate_splits(train=["a", "b", "c"], validation=["b", "d"])
    assert result.kept_train == [0, 2]
    assert result.kept_validation == [0, 1]
    assert result.train_dropped_as_in_validation == 1


def test_repeats_within_a_split_keep_the_first() -> None:
    result = deduplicate_splits(train=["a", "a", "b", "a"], validation=["c", "c"])
    assert result.kept_train == [0, 2]
    assert result.kept_validation == [0]
    assert result.train_dropped_as_repeat == 2
    assert result.validation_dropped_as_repeat == 1
