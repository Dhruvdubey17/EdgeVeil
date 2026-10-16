import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from edgeveil.bench.environment import environment_record
from edgeveil.detector.alignment import is_special_token
from edgeveil.detector.builds import (
    Backend,
    BuildRecord,
    ComputeUnits,
    Precision,
    build_record,
    load_detector,
)
from edgeveil.detector.inference import WindowLogits
from edgeveil.detector.labels import OUTSIDE_ID
from edgeveil.detector.torch_backend import TorchDetector
from edgeveil.detector.windows import Window
from edgeveil.export.calibration import (
    PARITY_SAMPLE_COUNT,
    SAMPLED_ROW_LIMIT,
    VALIDATION_SAMPLE_SEED,
    validation_sample,
)

RESULT_DECIMALS = 6
REFERENCE_DESCRIPTION = "torch fp32 on CPU, same tokenizer and windows"
AGREEMENT_DEFINITION = (
    "share of non-special tokens whose highest-scoring BIO label matches the "
    "reference; entity agreement counts only tokens the reference labels as an "
    "entity"
)


@dataclass(frozen=True, slots=True)
class ParityReport:
    windows: int
    tokens: int
    token_agreement: float
    reference_entity_tokens: int
    entity_token_agreement: float
    max_abs_logit_difference: float
    mean_abs_logit_difference: float


def content_rows(window: Window) -> NDArray[np.bool_]:
    return np.array([not is_special_token(offset) for offset in window.offsets])


def compare_logits(
    reference: Sequence[WindowLogits],
    candidate: Sequence[WindowLogits],
    windows: Sequence[Window],
) -> ParityReport:
    rows = [content_rows(window) for window in windows]
    expected = np.concatenate(
        [logits[mask] for logits, mask in zip(reference, rows, strict=True)]
    )
    actual = np.concatenate(
        [logits[mask] for logits, mask in zip(candidate, rows, strict=True)]
    )
    expected_labels = expected.argmax(axis=1)
    agrees = expected_labels == actual.argmax(axis=1)
    is_entity = expected_labels != OUTSIDE_ID
    difference = np.abs(expected - actual)
    return ParityReport(
        windows=len(windows),
        tokens=len(expected_labels),
        token_agreement=float(agrees.mean()),
        reference_entity_tokens=int(is_entity.sum()),
        entity_token_agreement=float(agrees[is_entity].mean()),
        max_abs_logit_difference=float(difference.max()),
        mean_abs_logit_difference=float(difference.mean()),
    )


@dataclass(frozen=True, slots=True)
class MeasuredParity:
    report: ParityReport
    build: BuildRecord


def measure_parity(
    model_dir: Path,
    data_dir: Path,
    backend: Backend,
    precisions: Sequence[Precision],
    compute_units: ComputeUnits = ComputeUnits.ALL,
) -> dict[Precision, MeasuredParity]:
    reference = TorchDetector(model_dir, device=torch.device("cpu"))
    windows = validation_sample(data_dir, reference.tokenizer).parity
    expected = reference.window_logits(windows)
    measured: dict[Precision, MeasuredParity] = {}
    for precision in precisions:
        detector = load_detector(
            model_dir, backend, precision, compute_units=compute_units
        )
        measured[precision] = MeasuredParity(
            report=compare_logits(expected, detector.window_logits(windows), windows),
            build=build_record(model_dir, backend, precision, compute_units),
        )
    return measured


def write_parity_results(
    report: ParityReport, build: BuildRecord, model_sha256: str, path: Path
) -> None:
    measures = {
        key: round(value, RESULT_DECIMALS) if isinstance(value, float) else value
        for key, value in asdict(report).items()
    }
    payload = {
        "precision": build.precision,
        "backend": build.backend,
        "compute_units": build.compute_units,
        "build": asdict(build),
        "model_sha256": model_sha256,
        "reference": REFERENCE_DESCRIPTION,
        "agreement": AGREEMENT_DEFINITION,
        "sample": {
            "split": "validation",
            "seed": VALIDATION_SAMPLE_SEED,
            "windows": PARITY_SAMPLE_COUNT,
            "rows_considered": SAMPLED_ROW_LIMIT,
            "disjoint_from_calibration": True,
        },
        "parity": measures,
        "environment": environment_record(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
