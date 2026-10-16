import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from coremltools.converters.mil.mil import types
from transformers import (
    DistilBertConfig,
    DistilBertForTokenClassification,
    PreTrainedTokenizerFast,
)

from edgeveil.detector.builds import ComputeUnits, Precision
from edgeveil.detector.coreml_backend import CoreMLDetector
from edgeveil.detector.labels import ID_TO_LABEL, LABEL_TO_ID
from edgeveil.detector.torch_backend import TorchDetector
from edgeveil.detector.windows import Window, encode_windows
from edgeveil.export.coreml_builds import convert_fp16, quantize_weights
from edgeveil.export.coreml_ops import fill_dtype

TEXTS = (
    "Ada Quill mail ada@example.com now",
    " ".join(f"w{index % 40}" for index in range(200)),
    " ".join(f"w{index % 40}" for index in range(450)),
)
FP16_LOGIT_TOLERANCE = 0.02
TORCH_BOOL = 11

needs_macos = pytest.mark.skipif(
    sys.platform != "darwin", reason="Core ML predictions run only on macOS"
)


def test_new_ones_takes_the_source_dtype_unless_one_is_passed() -> None:
    source = SimpleNamespace(dtype=types.int32)
    assert fill_dtype(source, None) is np.int32
    assert fill_dtype(source, SimpleNamespace(val=None)) is np.int32
    assert fill_dtype(source, SimpleNamespace(val=torch.bool)) is bool
    assert fill_dtype(source, SimpleNamespace(val=TORCH_BOOL)) is bool
    assert fill_dtype(source, SimpleNamespace(val=torch.int64)) is np.int32


@pytest.fixture(scope="module")
def model_dir(
    tiny_tokenizer: PreTrainedTokenizerFast, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=len(tiny_tokenizer),
        dim=64,
        n_layers=1,
        n_heads=2,
        hidden_dim=128,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    path = tmp_path_factory.mktemp("tiny_coreml")
    DistilBertForTokenClassification(config).save_pretrained(path)
    tiny_tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def fp16_path(model_dir: Path) -> Path:
    path = model_dir / "fp16.mlpackage"
    convert_fp16(model_dir).save(str(path))
    return path


@pytest.fixture(scope="module")
def windows(model_dir: Path) -> list[Window]:
    detector = TorchDetector(model_dir, device=torch.device("cpu"))
    return [w for text in encode_windows(detector.tokenizer, TEXTS) for w in text]


@needs_macos
def test_padded_core_ml_logits_match_torch_at_every_enumerated_length(
    model_dir: Path, fp16_path: Path, windows: list[Window]
) -> None:
    reference = TorchDetector(model_dir, device=torch.device("cpu"))
    converted = CoreMLDetector(
        fp16_path, model_dir, compute_units=ComputeUnits.CPU_ONLY
    )
    assert sorted(len(window.input_ids) for window in windows)[-1] > 256
    for expected, actual in zip(
        reference.window_logits(windows), converted.window_logits(windows), strict=True
    ):
        assert np.abs(expected - actual).max() < FP16_LOGIT_TOLERANCE


@needs_macos
@pytest.mark.parametrize("precision", ["int8", "int4"])
def test_weight_quantized_packages_run(
    precision: str, model_dir: Path, fp16_path: Path, windows: list[Window]
) -> None:
    path = model_dir / f"{precision}.mlpackage"
    quantize_weights(fp16_path, Precision(precision)).save(str(path))
    detector = CoreMLDetector(path, model_dir, compute_units=ComputeUnits.CPU_ONLY)
    logits = detector.window_logits(windows)
    assert all(np.isfinite(row).all() for row in logits)
