import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset

from edgeveil.dataset.credentials import AugmentedSplit, LabeledText, augment_split
from edgeveil.dataset.openpii import OPENPII_SPLITS, DatasetError
from edgeveil.dataset.prepare import openpii_clean_dir
from edgeveil.spans import EntityType, GoldSpan

DATA_DIRECTORY_NAME = "data"
REPORT_NAME = "report.json"


class OutputOutsideDataError(DatasetError):
    pass


@dataclass(frozen=True, slots=True)
class AugmentationReport:
    seed: int
    rows_by_origin: dict[str, dict[str, int]]
    credential_spans: dict[str, int]
    credential_row_share: dict[str, float]


def openpii_augmented_dir(data_dir: Path) -> Path:
    return data_dir / "openpii" / "augmented"


def ensure_inside_data_directory(path: Path) -> None:
    if DATA_DIRECTORY_NAME not in path.resolve().parts:
        raise OutputOutsideDataError(
            f"credentials are written only under a {DATA_DIRECTORY_NAME}/ directory"
        )


def split_rng(seed: int, split: str) -> random.Random:
    return random.Random(f"{seed}:{split}")


def labeled_text(record: Mapping[str, Any]) -> LabeledText:
    spans = tuple(
        GoldSpan(span["start"], span["end"], EntityType(span["label"]))
        for span in record["spans"]
    )
    return LabeledText(text=record["source_text"], spans=spans)


def load_clean_split(data_dir: Path, split: str) -> tuple[list[int], list[LabeledText]]:
    clean_dir = openpii_clean_dir(data_dir)
    records = Dataset.from_parquet(
        str(clean_dir / f"{split}.parquet"), cache_dir=str(clean_dir / "cache")
    ).to_list()
    return [record["uid"] for record in records], [
        labeled_text(record) for record in records
    ]


def augmented_records(
    uids: Sequence[int], split: AugmentedSplit
) -> list[dict[str, Any]]:
    padded_uids: list[int | None] = [*uids, *([None] * (len(split.rows) - len(uids)))]
    return [
        {
            "uid": uid,
            "origin": origin.value,
            "source_text": row.text,
            "spans": [
                {"start": span.start, "end": span.end, "label": span.label.value}
                for span in row.spans
            ],
        }
        for uid, row, origin in zip(padded_uids, split.rows, split.origins, strict=True)
    ]


def has_credential(row: LabeledText) -> bool:
    return any(span.label is EntityType.CREDENTIAL for span in row.spans)


def count_credential_spans(rows: Sequence[LabeledText]) -> int:
    labels = [span.label for row in rows for span in row.spans]
    return labels.count(EntityType.CREDENTIAL)


def augment_and_write_split(data_dir: Path, split: str, seed: int) -> AugmentedSplit:
    uids, rows = load_clean_split(data_dir, split)
    augmented = augment_split(rows, split_rng(seed, split))
    path = openpii_augmented_dir(data_dir) / f"{split}.parquet"
    Dataset.from_list(augmented_records(uids, augmented)).to_parquet(str(path))
    return augmented


def augment_openpii(data_dir: Path, seed: int) -> AugmentationReport:
    output_dir = openpii_augmented_dir(data_dir)
    ensure_inside_data_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        split: augment_and_write_split(data_dir, split, seed)
        for split in OPENPII_SPLITS
    }
    report = AugmentationReport(
        seed=seed,
        rows_by_origin={
            split: dict(Counter(origin.value for origin in augmented.origins))
            for split, augmented in splits.items()
        },
        credential_spans={
            split: count_credential_spans(augmented.rows)
            for split, augmented in splits.items()
        },
        credential_row_share={
            split: sum(map(has_credential, augmented.rows)) / len(augmented.rows)
            for split, augmented in splits.items()
        },
    )
    (output_dir / REPORT_NAME).write_text(json.dumps(asdict(report), indent=2))
    return report
