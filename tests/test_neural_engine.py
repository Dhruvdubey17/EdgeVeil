from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForTokenClassification,
    PreTrainedTokenizerFast,
)

from edgeveil.dataset.augment import openpii_augmented_dir
from edgeveil.detector.coreml_backend import enumerated_length
from edgeveil.detector.inference import MASKED_POSITION_BIAS, neural_engine_inputs
from edgeveil.detector.labels import ID_TO_LABEL, LABEL_TO_ID
from edgeveil.detector.torch_backend import TorchDetector
from edgeveil.detector.windows import Window, encode_windows
from edgeveil.export.calibration import validation_sample
from edgeveil.export.neural_engine import (
    NeuralEngineTokenClassifier,
    load_neural_engine_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert"
DATA_DIR = PROJECT_ROOT / "data"
VALIDATION_PATH = openpii_augmented_dir(DATA_DIR) / "validation.parquet"
VALIDATION_WINDOW_COUNT = 200
EQUIVALENCE_TOLERANCE = 1e-4
TEXTS = (
    "Ada Quill mail ada@example.com now",
    " ".join(f"w{index % 40}" for index in range(200)),
    " ".join(f"w{index % 40}" for index in range(480)),
)


def neural_engine_logits(
    model: NeuralEngineTokenClassifier, windows: list[Window]
) -> list[np.ndarray]:
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for window in windows:
            token_count = len(window.input_ids)
            inputs = neural_engine_inputs(window, enumerated_length(token_count))
            output = model(
                torch.from_numpy(inputs.input_ids.astype(np.int64)),
                torch.from_numpy(inputs.position_ids.astype(np.int64)),
                torch.from_numpy(inputs.attention_bias),
            )
            logits.append(output[0, :, 0, :token_count].T.numpy())
    return logits


def largest_difference(model_dir: Path, windows: list[Window]) -> float:
    reference = TorchDetector(model_dir, device=torch.device("cpu"))
    converted = neural_engine_logits(load_neural_engine_model(model_dir), windows)
    return max(
        float(np.abs(expected - actual).max())
        for expected, actual in zip(
            reference.window_logits(windows), converted, strict=True
        )
    )


def test_inputs_mask_padding_and_number_positions() -> None:
    window = Window(input_ids=[101, 7, 102], offsets=[(0, 0), (0, 1), (0, 0)])
    inputs = neural_engine_inputs(window, 8)
    assert inputs.input_ids.shape == (1, 8)
    assert inputs.position_ids.tolist() == [list(range(8))]
    assert inputs.attention_bias.shape == (1, 8, 1, 1)
    assert inputs.attention_bias[0, :3, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert (inputs.attention_bias[0, 3:, 0, 0] == MASKED_POSITION_BIAS).all()


def test_neural_engine_layout_matches_the_hugging_face_model(
    tiny_tokenizer: PreTrainedTokenizerFast, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=len(tiny_tokenizer),
        dim=64,
        n_layers=2,
        n_heads=4,
        hidden_dim=128,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    DistilBertForTokenClassification(config).save_pretrained(tmp_path)
    tiny_tokenizer.save_pretrained(tmp_path)
    detector = TorchDetector(tmp_path, device=torch.device("cpu"))
    windows = [w for text in encode_windows(detector.tokenizer, TEXTS) for w in text]
    assert {enumerated_length(len(window.input_ids)) for window in windows} == {
        128,
        256,
        512,
    }
    assert largest_difference(tmp_path, windows) < EQUIVALENCE_TOLERANCE


@pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists() or not VALIDATION_PATH.exists(),
    reason="needs the local v1 model and the augmented validation split",
)
def test_neural_engine_layout_matches_v1_on_validation_windows() -> None:
    detector = TorchDetector(MODEL_DIR, device=torch.device("cpu"))
    windows = validation_sample(DATA_DIR, detector.tokenizer).parity
    sample = windows[:VALIDATION_WINDOW_COUNT]
    assert largest_difference(MODEL_DIR, sample) < EQUIVALENCE_TOLERANCE
