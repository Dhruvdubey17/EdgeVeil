from pathlib import Path

import pytest
from datasets import Dataset

from edgeveil.dataset.augment import (
    OutputOutsideDataError,
    augment_openpii,
    openpii_augmented_dir,
)
from edgeveil.dataset.credentials import Origin
from edgeveil.dataset.prepare import openpii_clean_dir

ROW_COUNT = 100


def write_clean_split(data_dir: Path, split: str) -> None:
    clean_dir = openpii_clean_dir(data_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "uid": index,
            "source_text": f"Row {index}. Mail ada@example.com today.",
            "name_joints": [],
            "spans": [
                {
                    "start": 11 + len(str(index)),
                    "end": 26 + len(str(index)),
                    "label": "EMAIL",
                }
            ],
        }
        for index in range(ROW_COUNT)
    ]
    Dataset.from_list(records).to_parquet(str(clean_dir / f"{split}.parquet"))


def test_augment_writes_both_splits_with_credentials(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for split in ("train", "validation"):
        write_clean_split(data_dir, split)

    report = augment_openpii(data_dir, seed=7)

    assert report.credential_spans["train"] > 0
    assert report.rows_by_origin["train"][Origin.OPENPII.value] < ROW_COUNT
    augmented = Dataset.from_parquet(
        str(openpii_augmented_dir(data_dir) / "train.parquet"),
        cache_dir=str(tmp_path / "cache"),
    ).to_list()
    emails = [
        record["source_text"][span["start"] : span["end"]]
        for record in augmented
        for span in record["spans"]
        if span["label"] == "EMAIL"
    ]
    assert emails
    assert all(
        email.startswith("ada@") and "." in email.split("@")[1] for email in emails
    )
    assert report.email_domains_rewritten["train"] > 0
    templates = [
        record
        for record in augmented
        if record["origin"] == Origin.CREDENTIAL_TEMPLATE.value
    ]
    assert all(record["uid"] is None for record in templates)


def test_augment_refuses_to_write_outside_a_data_directory(tmp_path: Path) -> None:
    with pytest.raises(OutputOutsideDataError):
        augment_openpii(tmp_path / "elsewhere", seed=7)


def write_person_split(data_dir: Path, split: str) -> None:
    clean_dir = openpii_clean_dir(data_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "uid": index,
            "source_text": "Ask Ada Quill now.",
            "spans": [{"start": 4, "end": 13, "label": "PERSON"}],
            "name_joints": [{"start": 7, "end": 8}],
        }
        for index in range(ROW_COUNT)
    ]
    Dataset.from_list(records).to_parquet(str(clean_dir / f"{split}.parquet"))


def test_augment_inserts_name_connectors_inside_person_spans(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for split in ("train", "validation"):
        write_person_split(data_dir, split)

    report = augment_openpii(data_dir, seed=7)

    assert report.name_pairs_eligible["train"] == ROW_COUNT
    assert report.name_connectors_inserted["train"] == 3
    augmented = Dataset.from_parquet(
        str(openpii_augmented_dir(data_dir) / "train.parquet"),
        cache_dir=str(tmp_path / "cache"),
    ).to_list()
    people = [
        record["source_text"][span["start"] : span["end"]]
        for record in augmented
        for span in record["spans"]
        if span["label"] == "PERSON"
    ]
    assert all(
        person.startswith("Ada ") and person.endswith(" Quill") for person in people
    )
    assert sum(person != "Ada Quill" for person in people) == 3
