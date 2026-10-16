import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForTokenClassification,
    PreTrainedTokenizerFast,
)

from edgeveil.detector.decoding import uniform_thresholds
from edgeveil.detector.labels import BIO_LABELS, ID_TO_LABEL, LABEL_TO_ID
from edgeveil.detector.onnx_backend import OnnxRuntimeDetector
from edgeveil.detector.protocol import BatchDetector
from edgeveil.detector.torch_backend import TorchDetector
from edgeveil.detector.windows import Window, encode_windows
from edgeveil.export.calibration import split_by_row
from edgeveil.export.onnx_builds import (
    convert_fp16,
    export_fp32,
    quantize_int4,
    quantize_int8_dynamic,
    quantize_int8_static,
)

TEXTS = (
    "Ada Quill mail ada@example.com now",
    "My password is w1 w2 w3",
    " ".join(f"w{index % 40}" for index in range(600)),
)
EXPORT_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def model_dir(
    tiny_tokenizer: PreTrainedTokenizerFast, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=len(tiny_tokenizer),
        dim=32,
        n_layers=1,
        n_heads=2,
        hidden_dim=64,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    path = tmp_path_factory.mktemp("tiny_export")
    DistilBertForTokenClassification(config).save_pretrained(path)
    tiny_tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def fp32_path(model_dir: Path) -> Path:
    path = model_dir / "onnx" / "fp32.onnx"
    path.parent.mkdir()
    export_fp32(model_dir, path)
    return path


@pytest.fixture(scope="module")
def windows(model_dir: Path) -> list[Window]:
    detector = TorchDetector(model_dir, device=torch.device("cpu"))
    return [w for text in encode_windows(detector.tokenizer, TEXTS) for w in text]


def test_onnx_logits_match_torch_at_every_window_length(
    model_dir: Path, fp32_path: Path, windows: list[Window]
) -> None:
    reference = TorchDetector(model_dir, device=torch.device("cpu"))
    exported = OnnxRuntimeDetector(fp32_path, model_dir)
    assert len({len(window.input_ids) for window in windows}) > 2
    for expected, actual in zip(
        reference.window_logits(windows), exported.window_logits(windows), strict=True
    ):
        assert np.abs(expected - actual).max() < EXPORT_TOLERANCE


def test_onnx_detector_finds_the_same_spans_as_torch(
    model_dir: Path, fp32_path: Path
) -> None:
    thresholds = uniform_thresholds(0.0)
    reference = TorchDetector(model_dir, thresholds, device=torch.device("cpu"))
    exported: BatchDetector = OnnxRuntimeDetector(fp32_path, model_dir, thresholds)
    expected = [span for spans in reference.detect_batch(TEXTS) for span in spans]
    actual = [span for spans in exported.detect_batch(TEXTS) for span in spans]
    assert expected
    assert [(s.start, s.end, s.label) for s in actual] == [
        (s.start, s.end, s.label) for s in expected
    ]
    assert [s.score for s in actual] == pytest.approx(
        [s.score for s in expected], abs=EXPORT_TOLERANCE
    )


def quantize_int8_static_on(windows: list[Window]) -> Callable[[Path, Path], None]:
    def quantize(source: Path, target: Path) -> None:
        quantize_int8_static(source, target, windows)

    return quantize


@pytest.mark.parametrize(
    ("build", "expected_op"),
    [
        ("fp16", "Cast"),
        ("int8-dynamic", "MatMulInteger"),
        ("int8-static", "QuantizeLinear"),
        ("int4", "MatMulNBits"),
    ],
)
def test_each_precision_builds_a_runnable_model(
    build: str,
    expected_op: str,
    model_dir: Path,
    fp32_path: Path,
    windows: list[Window],
) -> None:
    builders = {
        "fp16": convert_fp16,
        "int8-dynamic": quantize_int8_dynamic,
        "int8-static": quantize_int8_static_on(windows),
        "int4": quantize_int4,
    }
    path = fp32_path.parent / f"{build}.onnx"
    builders[build](fp32_path, path)
    ops = Counter(node.op_type for node in onnx.load(path).graph.node)
    assert ops[expected_op] > 0
    logits = OnnxRuntimeDetector(path, model_dir).window_logits(windows)
    assert [row.shape for row in logits] == [
        (len(window.input_ids), len(BIO_LABELS)) for window in windows
    ]
    assert all(np.isfinite(row).all() for row in logits)


def window_of(row: int, part: int) -> Window:
    return Window(input_ids=[row, part], offsets=[(0, 0), (0, 0)])


def test_calibration_and_parity_samples_never_share_a_row() -> None:
    per_row = [
        [window_of(row, part) for part in range(row % 3 + 1)] for row in range(40)
    ]
    sample = split_by_row(per_row, calibration_count=10, parity_count=12)
    calibration_rows = {window.input_ids[0] for window in sample.calibration}
    parity_rows = {window.input_ids[0] for window in sample.parity}
    assert len(sample.calibration) == 10
    assert len(sample.parity) == 12
    assert calibration_rows.isdisjoint(parity_rows)


def test_onnx_runtime_path_imports_without_torch_or_transformers() -> None:
    probe = (
        "import sys, edgeveil.detector.onnx_backend, edgeveil.detector.builds; "
        "print(sorted(name for name in ('torch', 'transformers') "
        "if name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
