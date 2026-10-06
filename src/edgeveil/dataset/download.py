import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from datasets import IterableDataset, load_dataset

from edgeveil.dataset.openpii import (
    ENGLISH_LANGUAGE_CODE,
    OPENPII_REPOSITORY,
    OPENPII_REVISION,
    OPENPII_SPLITS,
)

KEPT_COLUMNS = [
    "uid",
    "source_text",
    "privacy_mask",
    "language",
    "region",
    "source_dataset",
]
DOWNLOAD_RECORD_NAME = "download.json"


@dataclass(frozen=True, slots=True)
class DownloadRecord:
    repository: str
    revision: str
    downloaded_at: str
    language: str


def openpii_raw_dir(data_dir: Path) -> Path:
    return data_dir / "openpii" / "raw"


def raw_split_path(data_dir: Path, split: str) -> Path:
    return openpii_raw_dir(data_dir) / f"{split}.parquet"


def is_english(language: str) -> bool:
    return language == ENGLISH_LANGUAGE_CODE


def stream_english_rows(split: str) -> IterableDataset:
    stream = load_dataset(
        OPENPII_REPOSITORY, revision=OPENPII_REVISION, split=split, streaming=True
    )
    assert isinstance(stream, IterableDataset)
    return stream.select_columns(KEPT_COLUMNS).filter(
        is_english, input_columns="language"
    )


def download_english_subset(data_dir: Path) -> DownloadRecord:
    output_dir = openpii_raw_dir(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in OPENPII_SPLITS:
        stream_english_rows(split).to_parquet(raw_split_path(data_dir, split))
    record = DownloadRecord(
        repository=OPENPII_REPOSITORY,
        revision=OPENPII_REVISION,
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        language=ENGLISH_LANGUAGE_CODE,
    )
    (output_dir / DOWNLOAD_RECORD_NAME).write_text(json.dumps(asdict(record), indent=2))
    return record
