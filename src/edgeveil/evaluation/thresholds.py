import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from numpy.typing import NDArray

from edgeveil.bench.environment import environment_record
from edgeveil.dataset.augment import labeled_text, openpii_augmented_dir
from edgeveil.dataset.openpii import OPENPII_REVISION
from edgeveil.detector.decoding import logits_to_spans, uniform_thresholds
from edgeveil.detector.metrics import EntityScores
from edgeveil.detector.torch_backend import TorchDetector
from edgeveil.detector.windows import Window, encode_windows
from edgeveil.evaluation.scoring import content_offsets, score_emitted_spans
from edgeveil.spans import EntityType, GoldSpan, LabeledText

THRESHOLD_BETA = 2.0
THRESHOLD_GRID = tuple(step / 20 for step in range(1, 20))
TUNING_SPLIT = "validation"
RESULT_DECIMALS = 4


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    threshold: float
    precision: float
    recall: float
    f_beta: float
    support: int


@dataclass(frozen=True, slots=True)
class ThresholdReport:
    model_dir: str
    validation_rows: int
    curves: dict[EntityType, list[ThresholdPoint]]
    chosen: dict[EntityType, ThresholdPoint]


def f_beta(precision: float, recall: float, beta: float = THRESHOLD_BETA) -> float:
    weighted = beta**2 * precision + recall
    if weighted == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / weighted


def threshold_point(threshold: float, scores: EntityScores | None) -> ThresholdPoint:
    if scores is None:
        return ThresholdPoint(threshold, 0.0, 0.0, 0.0, support=0)
    return ThresholdPoint(
        threshold,
        scores.precision,
        scores.recall,
        f_beta(scores.precision, scores.recall),
        scores.support,
    )


def choose_thresholds(
    curves: Mapping[EntityType, Sequence[ThresholdPoint]],
) -> dict[EntityType, ThresholdPoint]:
    return {
        entity: max(points, key=lambda point: (point.f_beta, -point.threshold))
        for entity, points in curves.items()
    }


def load_tuning_rows(data_dir: Path, limit: int | None) -> list[LabeledText]:
    augmented_dir = openpii_augmented_dir(data_dir)
    rows = Dataset.from_parquet(
        str(augmented_dir / f"{TUNING_SPLIT}.parquet"),
        cache_dir=str(augmented_dir / "cache"),
    )
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    return [labeled_text(record) for record in rows.to_list()]


def per_text_logits(
    detector: TorchDetector, per_text_windows: Sequence[Sequence[Window]]
) -> list[list[NDArray[np.float32]]]:
    flat = [window for windows in per_text_windows for window in windows]
    flat_logits = detector.window_logits(flat)
    grouped: list[list[NDArray[np.float32]]] = []
    cursor = 0
    for windows in per_text_windows:
        grouped.append(flat_logits[cursor : cursor + len(windows)])
        cursor += len(windows)
    return grouped


def tune_thresholds(
    model_dir: Path,
    data_dir: Path,
    limit: int | None = None,
    device: torch.device | None = None,
) -> ThresholdReport:
    detector = TorchDetector(model_dir, device=device)
    rows = load_tuning_rows(data_dir, limit)
    texts = [row.text for row in rows]
    gold: list[Sequence[GoldSpan]] = [row.spans for row in rows]
    per_text_windows = encode_windows(detector.tokenizer, texts)
    logits = per_text_logits(detector, per_text_windows)
    offsets = content_offsets(detector.tokenizer, texts)
    curves: dict[EntityType, list[ThresholdPoint]] = {
        entity: [] for entity in EntityType
    }
    for threshold in THRESHOLD_GRID:
        thresholds = uniform_thresholds(threshold)
        emitted = [
            logits_to_spans(text_logits, windows, thresholds)
            for text_logits, windows in zip(logits, per_text_windows, strict=True)
        ]
        scores = score_emitted_spans(gold, emitted, offsets)
        for entity in EntityType:
            curves[entity].append(threshold_point(threshold, scores.get(entity.value)))
    return ThresholdReport(
        model_dir=str(model_dir),
        validation_rows=len(rows),
        curves=curves,
        chosen=choose_thresholds(curves),
    )


def rounded_point(point: ThresholdPoint) -> dict[str, Any]:
    return {
        key: round(value, RESULT_DECIMALS) if isinstance(value, float) else value
        for key, value in asdict(point).items()
    }


def training_record(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "training_run.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_threshold_results(report: ThresholdReport, path: Path) -> None:
    training = training_record(Path(report.model_dir))
    payload = {
        "model_dir": report.model_dir,
        "base_model": training.get("base_model"),
        "base_model_revision": training.get("base_model_revision"),
        "dataset_revision": OPENPII_REVISION,
        "split": TUNING_SPLIT,
        "validation_rows": report.validation_rows,
        "precision": "fp32",
        "backend": "torch",
        "beta": THRESHOLD_BETA,
        "grid": list(THRESHOLD_GRID),
        "chosen": {
            entity.value: rounded_point(point)
            for entity, point in report.chosen.items()
        },
        "curves": {
            entity.value: [rounded_point(point) for point in points]
            for entity, points in report.curves.items()
        },
        "environment": environment_record(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
