from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

from edgeveil.detector.decoding import Thresholds, logits_to_spans
from edgeveil.detector.windows import Window, encode_windows
from edgeveil.spans import Span

INFERENCE_BATCH_SIZE = 32
PADDING_TOKEN_ID = 0
INPUT_IDS_NAME = "input_ids"
ATTENTION_MASK_NAME = "attention_mask"
LOGITS_NAME = "logits"

WindowLogits = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class PaddedBatch:
    input_ids: NDArray[np.int64]
    attention_mask: NDArray[np.int64]


def padded_batch(windows: Sequence[Window], width: int | None = None) -> PaddedBatch:
    longest = max(len(window.input_ids) for window in windows)
    width = longest if width is None else width
    input_ids = np.full((len(windows), width), PADDING_TOKEN_ID, dtype=np.int64)
    attention_mask = np.zeros((len(windows), width), dtype=np.int64)
    for row, window in enumerate(windows):
        input_ids[row, : len(window.input_ids)] = window.input_ids
        attention_mask[row, : len(window.input_ids)] = 1
    return PaddedBatch(input_ids, attention_mask)


def batched_window_logits(
    windows: Sequence[Window],
    run_batch: Callable[[PaddedBatch], NDArray[np.floating]],
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> list[WindowLogits]:
    order = sorted(range(len(windows)), key=lambda index: len(windows[index].input_ids))
    logits: list[WindowLogits | None] = [None] * len(windows)
    for start in range(0, len(order), batch_size):
        batch_indices = order[start : start + batch_size]
        output = run_batch(padded_batch([windows[index] for index in batch_indices]))
        for row, index in enumerate(batch_indices):
            token_count = len(windows[index].input_ids)
            logits[index] = output[row, :token_count].astype(np.float32)
    return [row for row in logits if row is not None]


def grouped_by_text(
    flat_logits: Sequence[WindowLogits], per_text: Sequence[Sequence[Window]]
) -> list[list[WindowLogits]]:
    grouped: list[list[WindowLogits]] = []
    cursor = 0
    for windows in per_text:
        grouped.append(list(flat_logits[cursor : cursor + len(windows)]))
        cursor += len(windows)
    return grouped


def per_text_logits(
    per_text: Sequence[Sequence[Window]],
    window_logits: Callable[[Sequence[Window]], list[WindowLogits]],
) -> list[list[WindowLogits]]:
    flat = [window for windows in per_text for window in windows]
    return grouped_by_text(window_logits(flat), per_text)


def detect_texts(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    window_logits: Callable[[Sequence[Window]], list[WindowLogits]],
    thresholds: Thresholds,
) -> list[list[Span]]:
    per_text = encode_windows(tokenizer, texts)
    return [
        logits_to_spans(text_logits, windows, thresholds)
        for text_logits, windows in zip(
            per_text_logits(per_text, window_logits), per_text, strict=True
        )
    ]
