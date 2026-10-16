from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForTokenClassification

from edgeveil.detector.decoding import Thresholds, uniform_thresholds
from edgeveil.detector.inference import (
    PaddedBatch,
    WindowLogits,
    batched_window_logits,
    detect_texts,
)
from edgeveil.detector.windows import Window, load_tokenizer
from edgeveil.spans import Span


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchDetector:
    def __init__(
        self,
        model_dir: Path,
        thresholds: Thresholds | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or best_device()
        self.tokenizer = load_tokenizer(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.thresholds = thresholds or uniform_thresholds()

    def run_batch(self, batch: PaddedBatch) -> NDArray[np.float32]:
        with torch.inference_mode():
            output = self.model(
                input_ids=torch.from_numpy(batch.input_ids).to(self.device),
                attention_mask=torch.from_numpy(batch.attention_mask).to(self.device),
            )
        return output.logits.float().cpu().numpy()

    def window_logits(self, windows: Sequence[Window]) -> list[WindowLogits]:
        return batched_window_logits(windows, self.run_batch)

    def detect_batch(self, texts: Sequence[str]) -> list[list[Span]]:
        return detect_texts(self.tokenizer, texts, self.window_logits, self.thresholds)

    def detect(self, text: str) -> list[Span]:
        return self.detect_batch([text])[0]
