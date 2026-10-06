import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from edgeveil.dataset.download import openpii_raw_dir, raw_split_path
from edgeveil.dataset.prepare import (
    REPORT_NAME,
    clean_split,
    openpii_clean_dir,
    prepare_openpii,
)
from edgeveil.spans import EntityType, GoldSpan

SOURCE = "openpii-test"


def record(uid: int, text: str, *entities: tuple[str, str]) -> dict[str, Any]:
    mask = [
        {
            "value": value,
            "start": text.index(value),
            "end": text.index(value) + len(value),
            "label": label,
        }
        for label, value in entities
    ]
    return {
        "uid": uid,
        "source_text": text,
        "privacy_mask": mask,
        "source_dataset": SOURCE,
    }


GOOD = record(1, "Mail ada@example.com now.", ("EMAIL", "ada@example.com"))
SHIFTED = {
    **record(2, "Call 555-0100 today.", ("TELEPHONENUM", "555-0100")),
    "privacy_mask": [
        {"value": "555-0100", "start": 4, "end": 12, "label": "TELEPHONENUM"}
    ],
}
OVERLAPPING = {
    **record(3, "abcdefgh"),
    "privacy_mask": [
        {"value": "abcde", "start": 0, "end": 5, "label": "EMAIL"},
        {"value": "defgh", "start": 3, "end": 8, "label": "TELEPHONENUM"},
    ],
}

OFF_CARD = record(5, "Meet at 10:20am.", ("TIME", "10:20am"))


def test_clean_split_keeps_valid_rows_and_counts_dropped_ones() -> None:
    cleaning = clean_split([GOOD, SHIFTED, OVERLAPPING, OFF_CARD])
    assert [row.uid for row in cleaning.rows] == [1]
    assert cleaning.rows[0].spans == (GoldSpan(5, 20, EntityType.EMAIL),)
    assert cleaning.raw_rows == 4
    assert cleaning.rows_with_off_card_labels == 1
    assert cleaning.off_card_labels == {"TIME": 1}
    assert cleaning.rows_with_offset_mismatch == 1
    assert cleaning.mismatched_entities_by_label == {"TELEPHONENUM": 1}
    assert cleaning.rows_with_overlapping_entities == 1
    assert cleaning.raw_rows_by_source_dataset == {SOURCE: 4}


def test_clean_row_repr_hides_text() -> None:
    [row] = clean_split([GOOD]).rows
    assert "example.com" not in repr(row)


def write_raw_split(data_dir: Path, split: str, rows: list[dict[str, Any]]) -> None:
    openpii_raw_dir(data_dir).mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(raw_split_path(data_dir, split)))


def test_prepare_writes_deduplicated_clean_splits_and_a_report(tmp_path: Path) -> None:
    shared = record(4, "Write to bo@example.com please.", ("EMAIL", "bo@example.com"))
    write_raw_split(tmp_path, "train", [GOOD, shared, SHIFTED])
    write_raw_split(tmp_path, "validation", [shared])

    report = prepare_openpii(tmp_path)

    assert report.kept_rows == {"train": 1, "validation": 1}
    assert report.train_dropped_as_in_validation == 1
    assert report.rows_with_offset_mismatch == {"train": 1, "validation": 0}
    assert report.spans_by_type == {"train": {"EMAIL": 1}, "validation": {"EMAIL": 1}}
    clean_train = Dataset.from_parquet(
        str(openpii_clean_dir(tmp_path) / "train.parquet"),
        cache_dir=str(tmp_path / "cache"),
    )
    assert clean_train["uid"] == [1]
    assert clean_train["spans"] == [[{"start": 5, "end": 20, "label": "EMAIL"}]]
    saved = json.loads((openpii_clean_dir(tmp_path) / REPORT_NAME).read_text())
    assert saved["kept_rows"] == report.kept_rows
