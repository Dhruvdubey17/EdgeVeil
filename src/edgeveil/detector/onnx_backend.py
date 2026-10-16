from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime
from numpy.typing import NDArray

from edgeveil.detector.decoding import Thresholds, uniform_thresholds
from edgeveil.detector.inference import (
    ATTENTION_MASK_NAME,
    INPUT_IDS_NAME,
    LOGITS_NAME,
    PaddedBatch,
    WindowLogits,
    batched_window_logits,
    detect_texts,
)
from edgeveil.detector.windows import Window, load_tokenizer
from edgeveil.spans import Span

CPU_PROVIDER = "CPUExecutionProvider"


class OnnxRuntimeDetector:
    def __init__(
        self,
        model_path: Path,
        tokenizer_dir: Path,
        thresholds: Thresholds | None = None,
    ) -> None:
        self.tokenizer = load_tokenizer(tokenizer_dir)
        self.session = onnxruntime.InferenceSession(
            str(model_path), providers=[CPU_PROVIDER]
        )
        self.thresholds = thresholds or uniform_thresholds()

    def run_batch(self, batch: PaddedBatch) -> NDArray[np.floating]:
        feeds = {
            INPUT_IDS_NAME: batch.input_ids,
            ATTENTION_MASK_NAME: batch.attention_mask,
        }
        (logits,) = self.session.run([LOGITS_NAME], feeds)
        return np.asarray(logits)

    def window_logits(self, windows: Sequence[Window]) -> list[WindowLogits]:
        return batched_window_logits(windows, self.run_batch)

    def detect_batch(self, texts: Sequence[str]) -> list[list[Span]]:
        return detect_texts(self.tokenizer, texts, self.window_logits, self.thresholds)

    def detect(self, text: str) -> list[Span]:
        return self.detect_batch([text])[0]
