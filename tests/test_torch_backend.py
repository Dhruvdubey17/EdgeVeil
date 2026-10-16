from pathlib import Path

import pytest
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForTokenClassification,
    PreTrainedTokenizerFast,
)

from edgeveil.detector.decoding import uniform_thresholds
from edgeveil.detector.labels import BIO_LABELS, ID_TO_LABEL, LABEL_TO_ID
from edgeveil.detector.protocol import Detector
from edgeveil.detector.torch_backend import TorchDetector
from edgeveil.detector.windows import encode_windows
from edgeveil.spans import Span

TEXT = "Ada Quill mail ada@example.com now"


@pytest.fixture(scope="module")
def model_dir(
    tiny_tokenizer: PreTrainedTokenizerFast, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=len(tiny_tokenizer),
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    path = tmp_path_factory.mktemp("tiny_model")
    DistilBertForTokenClassification(config).save_pretrained(path)
    tiny_tokenizer.save_pretrained(path)
    return path


def test_torch_detector_satisfies_the_detector_protocol(model_dir: Path) -> None:
    detector: Detector = TorchDetector(
        model_dir, uniform_thresholds(0.0), device=torch.device("cpu")
    )
    spans = detector.detect(TEXT)
    assert spans
    assert all(isinstance(span, Span) for span in spans)
    assert all(0 <= span.start < span.end <= len(TEXT) for span in spans)


def test_window_logits_have_one_row_per_token(
    model_dir: Path, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    detector = TorchDetector(model_dir, device=torch.device("cpu"))
    windows = encode_windows(tiny_tokenizer.backend_tokenizer, [TEXT, "Ada"])
    flat = [window for text_windows in windows for window in text_windows]
    logits = detector.window_logits(flat)
    assert [row.shape for row in logits] == [
        (len(window.input_ids), len(BIO_LABELS)) for window in flat
    ]


def test_batched_logits_match_single_window_logits(
    model_dir: Path, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    detector = TorchDetector(model_dir, device=torch.device("cpu"))
    windows = [
        window
        for text in encode_windows(tiny_tokenizer.backend_tokenizer, [TEXT, "Ada now"])
        for window in text
    ]
    together = detector.window_logits(windows)
    alone = [detector.window_logits([window])[0] for window in windows]
    for batched, single in zip(together, alone, strict=True):
        assert batched == pytest.approx(single, abs=1e-5)


def test_threshold_of_one_finds_nothing(model_dir: Path) -> None:
    detector = TorchDetector(
        model_dir, uniform_thresholds(1.01), device=torch.device("cpu")
    )
    assert detector.detect(TEXT) == []
