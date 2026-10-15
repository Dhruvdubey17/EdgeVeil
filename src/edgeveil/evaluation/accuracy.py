import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from edgeveil.bench.environment import environment_record
from edgeveil.detector.torch_backend import TorchDetector
from edgeveil.evaluation.bootstrap import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    ScoredType,
    score_with_intervals,
)
from edgeveil.evaluation.scoring import content_offsets, iob2_tags
from edgeveil.evaluation.test_set import load_test_set
from edgeveil.spans import EntityType

RESULT_DECIMALS = 4
TRAINING_RECORD_NAME = "training_run.json"


class UnsupportedConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AccuracyReport:
    model_dir: str
    test_set_path: str
    test_set_sha256: str
    snippets: int
    spans_by_type: dict[str, int]
    thresholds_path: str
    thresholds: dict[str, float]
    resamples: int
    scores: dict[str, ScoredType]


def load_thresholds(path: Path) -> dict[EntityType, float]:
    chosen = json.loads(path.read_text())["chosen"]
    missing = [entity.value for entity in EntityType if entity.value not in chosen]
    if missing:
        raise UnsupportedConfigurationError(
            f"thresholds file lacks {', '.join(missing)}"
        )
    return {entity: float(chosen[entity.value]["threshold"]) for entity in EntityType}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_test_set(
    model_dir: Path,
    test_set_path: Path,
    thresholds_path: Path,
    resamples: int = BOOTSTRAP_RESAMPLES,
    device: torch.device | None = None,
) -> AccuracyReport:
    snippets = load_test_set(test_set_path)
    thresholds = load_thresholds(thresholds_path)
    detector = TorchDetector(model_dir, thresholds, device=device)
    texts = [snippet.text for snippet in snippets]
    emitted = detector.detect_batch(texts)
    offsets = content_offsets(detector.tokenizer, texts)
    gold_tags = [
        iob2_tags(snippet.spans, text_offsets)
        for snippet, text_offsets in zip(snippets, offsets, strict=True)
    ]
    emitted_tags = [
        iob2_tags(spans, text_offsets)
        for spans, text_offsets in zip(emitted, offsets, strict=True)
    ]
    spans_by_type = Counter(
        span.label.value for snippet in snippets for span in snippet.spans
    )
    return AccuracyReport(
        model_dir=str(model_dir),
        test_set_path=str(test_set_path),
        test_set_sha256=file_sha256(test_set_path),
        snippets=len(snippets),
        spans_by_type=dict(sorted(spans_by_type.items())),
        thresholds_path=str(thresholds_path),
        thresholds={entity.value: value for entity, value in thresholds.items()},
        resamples=resamples,
        scores=score_with_intervals(gold_tags, emitted_tags, resamples),
    )


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, RESULT_DECIMALS)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def training_record(model_dir: Path) -> dict[str, Any]:
    path = model_dir / TRAINING_RECORD_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_accuracy_results(report: AccuracyReport, path: Path) -> None:
    training = training_record(Path(report.model_dir))
    payload = {
        "model_dir": report.model_dir,
        "base_model": training.get("base_model"),
        "base_model_revision": training.get("base_model_revision"),
        "training_dataset_revision": training.get("dataset_revision"),
        "trained_on": training.get("accelerator"),
        "precision": "fp32",
        "backend": "torch",
        "test_set": {
            "path": report.test_set_path,
            "sha256": report.test_set_sha256,
            "snippets": report.snippets,
            "spans_by_type": report.spans_by_type,
        },
        "thresholds_file": report.thresholds_path,
        "thresholds": report.thresholds,
        "scoring": "final emitted spans re-encoded as IOB2, seqeval strict mode",
        "bootstrap": {
            "unit": "snippet",
            "resamples": report.resamples,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "scores": {
            name: rounded(asdict(scored)) for name, scored in report.scores.items()
        },
        "environment": environment_record(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
