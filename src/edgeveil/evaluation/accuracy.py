import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from edgeveil.bench.environment import environment_record
from edgeveil.detector.alignment import Offset
from edgeveil.detector.torch_backend import TorchDetector
from edgeveil.evaluation.bootstrap import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    ScoredType,
    score_with_intervals,
)
from edgeveil.evaluation.detection import Detection
from edgeveil.evaluation.errors import (
    DetectionError,
    detection_errors,
    error_counts,
    token_aligned,
)
from edgeveil.evaluation.scoring import content_offsets, iob2_tags
from edgeveil.evaluation.secondary import (
    SecondaryScores,
    match_counts,
    secondary_scores,
)
from edgeveil.evaluation.test_set import LabeledSnippet, load_test_set
from edgeveil.spans import EntityType, GoldSpan, Span
from edgeveil.validators.combiner import combine_spans
from edgeveil.validators.finders import find_validator_spans

RESULT_DECIMALS = 4
EXAMPLE_CONTEXT_CHARACTERS = 30
TRAINING_RECORD_NAME = "training_run.json"
MODEL_WEIGHTS_NAME = "model.safetensors"
SECONDARY_SCORING = {
    "overlap": "a span matches if it overlaps a span of the same type",
    "char_recall": "share of gold span characters covered by any emitted span",
}


class UnsupportedConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunInfo:
    model_dir: str
    model_sha256: str
    test_set_path: str
    test_set_sha256: str
    snippets: int
    spans_by_type: dict[str, int]
    thresholds_path: str
    thresholds_sha256: str
    thresholds: dict[str, float]
    resamples: int


@dataclass(frozen=True, slots=True)
class AccuracyReport:
    run: RunInfo
    detection: Detection
    scores: dict[str, ScoredType]
    secondary: dict[str, SecondaryScores]
    errors: list[DetectionError]


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


def score_detection(
    run: RunInfo,
    detection: Detection,
    snippets: Sequence[LabeledSnippet],
    emitted: Sequence[Sequence[Span]],
    offsets: Sequence[Sequence[Offset]],
) -> AccuracyReport:
    gold = [snippet.spans for snippet in snippets]
    gold_aligned = [
        token_aligned(spans, text) for spans, text in zip(gold, offsets, strict=True)
    ]
    emitted_aligned = [
        token_aligned(spans, text) for spans, text in zip(emitted, offsets, strict=True)
    ]
    gold_tags = [
        iob2_tags(spans, text) for spans, text in zip(gold, offsets, strict=True)
    ]
    emitted_tags = [
        iob2_tags(spans, text) for spans, text in zip(emitted, offsets, strict=True)
    ]
    counts = match_counts(gold_aligned, emitted_aligned, gold, emitted)
    return AccuracyReport(
        run=run,
        detection=detection,
        scores=score_with_intervals(gold_tags, emitted_tags, run.resamples),
        secondary=secondary_scores(counts, run.resamples),
        errors=detection_errors(gold_aligned, emitted_aligned),
    )


def evaluate_test_set(
    model_dir: Path,
    test_set_path: Path,
    thresholds_path: Path,
    resamples: int = BOOTSTRAP_RESAMPLES,
    device: torch.device | None = None,
) -> dict[Detection, AccuracyReport]:
    snippets = load_test_set(test_set_path)
    thresholds = load_thresholds(thresholds_path)
    detector = TorchDetector(model_dir, thresholds, device=device)
    texts = [snippet.text for snippet in snippets]
    model_spans = detector.detect_batch(texts)
    validator_spans = [find_validator_spans(text) for text in texts]
    emitted = {
        Detection.MODEL: model_spans,
        Detection.VALIDATORS: validator_spans,
        Detection.COMBINED: [
            combine_spans(text, from_model, from_validators)
            for text, from_model, from_validators in zip(
                texts, model_spans, validator_spans, strict=True
            )
        ],
    }
    offsets = content_offsets(detector.tokenizer, texts)
    spans_by_type = Counter(
        span.label.value for snippet in snippets for span in snippet.spans
    )
    run = RunInfo(
        model_dir=str(model_dir),
        model_sha256=file_sha256(model_dir / MODEL_WEIGHTS_NAME),
        test_set_path=str(test_set_path),
        test_set_sha256=file_sha256(test_set_path),
        snippets=len(snippets),
        spans_by_type=dict(sorted(spans_by_type.items())),
        thresholds_path=str(thresholds_path),
        thresholds_sha256=file_sha256(thresholds_path),
        thresholds={entity.value: value for entity, value in thresholds.items()},
        resamples=resamples,
    )
    return {
        detection: score_detection(run, detection, snippets, spans, offsets)
        for detection, spans in emitted.items()
    }


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
    run = report.run
    training = training_record(Path(run.model_dir))
    payload = {
        "detection": report.detection.value,
        "preset": Path(run.model_dir).name,
        "model_dir": run.model_dir,
        "model_sha256": run.model_sha256,
        "thresholds_sha256": run.thresholds_sha256,
        "base_model": training.get("base_model"),
        "base_model_revision": training.get("base_model_revision"),
        "training_dataset_revision": training.get("dataset_revision"),
        "trained_on": training.get("accelerator"),
        "precision": "fp32",
        "backend": "torch",
        "test_set": {
            "path": run.test_set_path,
            "sha256": run.test_set_sha256,
            "snippets": run.snippets,
            "spans_by_type": run.spans_by_type,
        },
        "thresholds_file": run.thresholds_path,
        "thresholds": run.thresholds,
        "scoring": "final emitted spans re-encoded as IOB2, seqeval strict mode",
        "secondary_scoring": SECONDARY_SCORING,
        "bootstrap": {
            "unit": "snippet",
            "resamples": run.resamples,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "scores": {
            name: rounded(asdict(scored)) for name, scored in report.scores.items()
        },
        "secondary_scores": {
            name: rounded(asdict(scored)) for name, scored in report.secondary.items()
        },
        "errors": error_counts(report.errors),
        "environment": environment_record(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def bracketed(text: str, span: GoldSpan) -> str:
    start = max(0, span.start - EXAMPLE_CONTEXT_CHARACTERS)
    end = min(len(text), span.end + EXAMPLE_CONTEXT_CHARACTERS)
    marked = (
        f"{text[start : span.start]}[{text[span.start : span.end]}]"
        f"{text[span.end : end]}"
    )
    return " ".join(marked.split())


def example_line(error: DetectionError, snippet: LabeledSnippet) -> str:
    heading = (
        f"{error.side.value} {error.span.label.value} {error.kind.value} "
        f"(snippet {snippet.snippet_id})"
    )
    line = f"{heading}: {bracketed(snippet.text, error.span)}"
    if error.counterpart is None:
        return line
    other = error.counterpart
    return f"{line} | other: {other.label.value} {bracketed(snippet.text, other)}"


def error_example_lines(report: AccuracyReport) -> list[str]:
    snippets = load_test_set(Path(report.run.test_set_path))
    ordered = sorted(
        report.errors,
        key=lambda error: (
            error.side,
            error.span.label,
            error.kind,
            error.snippet_index,
        ),
    )
    return [example_line(error, snippets[error.snippet_index]) for error in ordered]
