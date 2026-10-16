import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from onnxruntime.quantization import CalibrationDataReader
from tokenizers import Tokenizer

from edgeveil.detector.inference import (
    ATTENTION_MASK_NAME,
    INPUT_IDS_NAME,
    padded_batch,
)
from edgeveil.detector.windows import Window, encode_windows
from edgeveil.evaluation.thresholds import load_tuning_rows

VALIDATION_SAMPLE_SEED = 20261016
CALIBRATION_SAMPLE_COUNT = 300
PARITY_SAMPLE_COUNT = 2000
SAMPLED_ROW_LIMIT = 4000


@dataclass(frozen=True, slots=True)
class ValidationSample:
    calibration: list[Window]
    parity: list[Window]


def shuffled_validation_texts(data_dir: Path) -> list[str]:
    texts = [row.text for row in load_tuning_rows(data_dir, None)]
    random.Random(VALIDATION_SAMPLE_SEED).shuffle(texts)
    return texts[:SAMPLED_ROW_LIMIT]


def split_by_row(
    per_row: Sequence[Sequence[Window]], calibration_count: int, parity_count: int
) -> ValidationSample:
    calibration: list[Window] = []
    parity: list[Window] = []
    for windows in per_row:
        if len(calibration) < calibration_count:
            calibration.extend(windows)
        elif len(parity) < parity_count:
            parity.extend(windows)
    return ValidationSample(calibration[:calibration_count], parity[:parity_count])


def validation_sample(data_dir: Path, tokenizer: Tokenizer) -> ValidationSample:
    per_row = encode_windows(tokenizer, shuffled_validation_texts(data_dir))
    return split_by_row(per_row, CALIBRATION_SAMPLE_COUNT, PARITY_SAMPLE_COUNT)


def window_feeds(window: Window) -> dict[str, NDArray[np.int64]]:
    batch = padded_batch([window])
    return {INPUT_IDS_NAME: batch.input_ids, ATTENTION_MASK_NAME: batch.attention_mask}


class WindowCalibrationReader(CalibrationDataReader):
    def __init__(self, windows: Sequence[Window]) -> None:
        self.feeds: Iterator[dict[str, NDArray[np.int64]]] = (
            window_feeds(window) for window in windows
        )

    def get_next(self) -> dict[str, NDArray[np.int64]]:
        return next(self.feeds, {})
