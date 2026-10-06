import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset

from edgeveil.dataset.download import raw_split_path
from edgeveil.dataset.openpii import (
    OPENPII_REVISION,
    OPENPII_SPLITS,
    OverlappingEntitiesError,
    SplitDeduplication,
    deduplicate_splits,
    map_entities,
    mismatched_labels,
    parse_privacy_mask,
)
from edgeveil.spans import GoldSpan

REPORT_NAME = "report.json"


@dataclass(frozen=True, slots=True)
class CleanRow:
    uid: int
    source_text: str = field(repr=False)
    spans: tuple[GoldSpan, ...]


@dataclass(slots=True)
class SplitCleaning:
    rows: list[CleanRow] = field(default_factory=list[CleanRow])
    raw_rows: int = 0
    rows_with_offset_mismatch: int = 0
    rows_with_overlapping_entities: int = 0
    mismatched_entities_by_label: Counter[str] = field(default_factory=Counter[str])
    raw_rows_by_source_dataset: Counter[str] = field(default_factory=Counter[str])


@dataclass(frozen=True, slots=True)
class PreparationReport:
    revision: str
    raw_rows: dict[str, int]
    raw_rows_by_source_dataset: dict[str, int]
    rows_with_offset_mismatch: dict[str, int]
    mismatched_entities_by_label: dict[str, int]
    rows_with_overlapping_entities: dict[str, int]
    train_dropped_as_in_validation: int
    train_dropped_as_repeat: int
    validation_dropped_as_repeat: int
    kept_rows: dict[str, int]
    spans_by_type: dict[str, dict[str, int]]


def openpii_clean_dir(data_dir: Path) -> Path:
    return data_dir / "openpii" / "clean"


def clean_split(records: Iterable[Mapping[str, Any]]) -> SplitCleaning:
    cleaning = SplitCleaning()
    for record in records:
        cleaning.raw_rows += 1
        cleaning.raw_rows_by_source_dataset[str(record["source_dataset"])] += 1
        add_clean_row(cleaning, record)
    return cleaning


def add_clean_row(cleaning: SplitCleaning, record: Mapping[str, Any]) -> None:
    text = record["source_text"]
    entities = parse_privacy_mask(record["privacy_mask"])
    mismatches = mismatched_labels(text, entities)
    if mismatches:
        cleaning.rows_with_offset_mismatch += 1
        cleaning.mismatched_entities_by_label.update(mismatches)
        return
    try:
        spans = map_entities(text, entities)
    except OverlappingEntitiesError:
        cleaning.rows_with_overlapping_entities += 1
        return
    cleaning.rows.append(
        CleanRow(uid=record["uid"], source_text=text, spans=tuple(spans))
    )


def count_spans_by_type(rows: Sequence[CleanRow]) -> dict[str, int]:
    counts = Counter(span.label.value for row in rows for span in row.spans)
    return dict(sorted(counts.items()))


def row_record(row: CleanRow) -> dict[str, Any]:
    return {
        "uid": row.uid,
        "source_text": row.source_text,
        "spans": [
            {"start": span.start, "end": span.end, "label": span.label.value}
            for span in row.spans
        ],
    }


def load_raw_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = raw_split_path(data_dir, split)
    cache_dir = path.parent / "cache"
    return Dataset.from_parquet(str(path), cache_dir=str(cache_dir)).to_list()


def write_clean_split(data_dir: Path, split: str, rows: Sequence[CleanRow]) -> None:
    path = openpii_clean_dir(data_dir) / f"{split}.parquet"
    Dataset.from_list([row_record(row) for row in rows]).to_parquet(str(path))


def merged_counts(counters: Iterable[Counter[str]]) -> dict[str, int]:
    return dict(sorted(sum(counters, Counter[str]()).items()))


def per_split(
    cleanings: Mapping[str, SplitCleaning], count: Callable[[SplitCleaning], int]
) -> dict[str, int]:
    return {split: count(cleaning) for split, cleaning in cleanings.items()}


def deduplicate(
    cleanings: Mapping[str, SplitCleaning],
) -> tuple[dict[str, list[CleanRow]], SplitDeduplication]:
    train, validation = cleanings["train"].rows, cleanings["validation"].rows
    deduplication = deduplicate_splits(
        train=[row.source_text for row in train],
        validation=[row.source_text for row in validation],
    )
    kept = {
        "train": [train[index] for index in deduplication.kept_train],
        "validation": [validation[index] for index in deduplication.kept_validation],
    }
    return kept, deduplication


def build_report(
    cleanings: Mapping[str, SplitCleaning],
    deduplication: SplitDeduplication,
    kept: Mapping[str, Sequence[CleanRow]],
) -> PreparationReport:
    return PreparationReport(
        revision=OPENPII_REVISION,
        raw_rows=per_split(cleanings, lambda cleaning: cleaning.raw_rows),
        raw_rows_by_source_dataset=merged_counts(
            cleaning.raw_rows_by_source_dataset for cleaning in cleanings.values()
        ),
        rows_with_offset_mismatch=per_split(
            cleanings, lambda cleaning: cleaning.rows_with_offset_mismatch
        ),
        mismatched_entities_by_label=merged_counts(
            cleaning.mismatched_entities_by_label for cleaning in cleanings.values()
        ),
        rows_with_overlapping_entities=per_split(
            cleanings, lambda cleaning: cleaning.rows_with_overlapping_entities
        ),
        train_dropped_as_in_validation=deduplication.train_dropped_as_in_validation,
        train_dropped_as_repeat=deduplication.train_dropped_as_repeat,
        validation_dropped_as_repeat=deduplication.validation_dropped_as_repeat,
        kept_rows={split: len(rows) for split, rows in kept.items()},
        spans_by_type={
            split: count_spans_by_type(rows) for split, rows in kept.items()
        },
    )


def prepare_openpii(data_dir: Path) -> PreparationReport:
    cleanings = {
        split: clean_split(load_raw_split(data_dir, split)) for split in OPENPII_SPLITS
    }
    kept, deduplication = deduplicate(cleanings)
    openpii_clean_dir(data_dir).mkdir(parents=True, exist_ok=True)
    for split, rows in kept.items():
        write_clean_split(data_dir, split, rows)
    report = build_report(cleanings, deduplication, kept)
    report_path = openpii_clean_dir(data_dir) / REPORT_NAME
    report_path.write_text(json.dumps(asdict(report), indent=2))
    return report
