from collections.abc import Sequence
from pathlib import Path

import coremltools
import numpy as np

from edgeveil.detector.builds import ANE_SEQUENCE_LENGTHS, ComputeUnits
from edgeveil.detector.decoding import Thresholds, uniform_thresholds
from edgeveil.detector.inference import (
    ATTENTION_BIAS_NAME,
    ATTENTION_MASK_NAME,
    INPUT_IDS_NAME,
    LOGITS_NAME,
    POSITION_IDS_NAME,
    WindowLogits,
    detect_texts,
    neural_engine_inputs,
    padded_batch,
)
from edgeveil.detector.windows import Window, load_tokenizer
from edgeveil.spans import Span

COREML_COMPUTE_UNITS = {
    ComputeUnits.ALL: coremltools.ComputeUnit.ALL,
    ComputeUnits.CPU_AND_NE: coremltools.ComputeUnit.CPU_AND_NE,
    ComputeUnits.CPU_AND_GPU: coremltools.ComputeUnit.CPU_AND_GPU,
    ComputeUnits.CPU_ONLY: coremltools.ComputeUnit.CPU_ONLY,
}


class WindowTooLongError(ValueError):
    pass


def enumerated_length(token_count: int) -> int:
    for length in ANE_SEQUENCE_LENGTHS:
        if token_count <= length:
            return length
    raise WindowTooLongError(
        f"window of {token_count} tokens exceeds {max(ANE_SEQUENCE_LENGTHS)}"
    )


class CoreMLDetector:
    def __init__(
        self,
        package_path: Path,
        tokenizer_dir: Path,
        thresholds: Thresholds | None = None,
        compute_units: ComputeUnits = ComputeUnits.ALL,
    ) -> None:
        self.tokenizer = load_tokenizer(tokenizer_dir)
        self.model = coremltools.models.MLModel(
            str(package_path), compute_units=COREML_COMPUTE_UNITS[compute_units]
        )
        self.thresholds = thresholds or uniform_thresholds()

    def single_window_logits(self, window: Window) -> WindowLogits:
        token_count = len(window.input_ids)
        batch = padded_batch([window], enumerated_length(token_count))
        output = self.model.predict(
            {
                INPUT_IDS_NAME: batch.input_ids.astype(np.int32),
                ATTENTION_MASK_NAME: batch.attention_mask.astype(np.int32),
            }
        )
        return np.asarray(output[LOGITS_NAME])[0, :token_count].astype(np.float32)

    def window_logits(self, windows: Sequence[Window]) -> list[WindowLogits]:
        return [self.single_window_logits(window) for window in windows]

    def detect_batch(self, texts: Sequence[str]) -> list[list[Span]]:
        return detect_texts(self.tokenizer, texts, self.window_logits, self.thresholds)

    def detect(self, text: str) -> list[Span]:
        return self.detect_batch([text])[0]


class NeuralEngineDetector(CoreMLDetector):
    def single_window_logits(self, window: Window) -> WindowLogits:
        token_count = len(window.input_ids)
        inputs = neural_engine_inputs(window, enumerated_length(token_count))
        output = self.model.predict(
            {
                INPUT_IDS_NAME: inputs.input_ids,
                POSITION_IDS_NAME: inputs.position_ids,
                ATTENTION_BIAS_NAME: inputs.attention_bias,
            }
        )
        channels_first = np.asarray(output[LOGITS_NAME])[0, :, 0, :token_count]
        return channels_first.T.astype(np.float32)
