from pathlib import Path

import pytest
from transformers import AutoTokenizer

from edgeveil.dataset.augment import openpii_augmented_dir
from edgeveil.detector.alignment import Offset
from edgeveil.detector.windows import (
    FRAME_PROBE_TEXT,
    MAX_WINDOW_TOKENS,
    WINDOW_OVERLAP_TOKENS,
    Window,
    WindowLayout,
    encode_windows,
    frame_windows,
    load_tokenizer,
)
from edgeveil.evaluation.thresholds import load_tuning_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert"
DATA_DIR = PROJECT_ROOT / "data"
VALIDATION_PATH = openpii_augmented_dir(DATA_DIR) / "validation.parquet"

WindowKey = tuple[tuple[int, ...], tuple[Offset, ...]]

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer.json").exists() or not VALIDATION_PATH.exists(),
    reason="needs the local v1 model and the augmented validation split",
)


def window_keys(windows: list[Window]) -> list[WindowKey]:
    return [(tuple(window.input_ids), tuple(window.offsets)) for window in windows]


def training_tokenizer_windows(texts: list[str]) -> list[list[WindowKey]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    framed = tokenizer(FRAME_PROBE_TEXT)["input_ids"]
    bare = tokenizer(FRAME_PROBE_TEXT, add_special_tokens=False)["input_ids"]
    text_start = framed.index(bare[0])
    layout = WindowLayout(
        prefix=framed[:text_start],
        suffix=framed[text_start + len(bare) :],
        max_tokens=MAX_WINDOW_TOKENS,
        overlap=WINDOW_OVERLAP_TOKENS,
    )
    encoded = tokenizer(
        texts, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )
    return [
        window_keys(frame_windows(text, ids, offsets, layout))
        for text, ids, offsets in zip(
            texts, encoded["input_ids"], encoded["offset_mapping"], strict=True
        )
    ]


def test_tokenizers_alone_match_the_training_tokenizer_on_validation() -> None:
    texts = [row.text for row in load_tuning_rows(DATA_DIR, None)]
    reference = training_tokenizer_windows(texts)
    runtime = [
        window_keys(windows)
        for windows in encode_windows(load_tokenizer(MODEL_DIR), texts)
    ]
    mismatched_rows = [
        index
        for index, (expected, actual) in enumerate(zip(reference, runtime, strict=True))
        if expected != actual
    ]
    window_count = sum(len(keys) for keys in runtime)
    assert len(texts) > 40_000
    assert window_count >= len(texts)
    assert mismatched_rows == []
