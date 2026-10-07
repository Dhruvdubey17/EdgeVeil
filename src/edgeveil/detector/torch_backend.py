from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForTokenClassification, AutoTokenizer

from edgeveil.detector.decoding import Thresholds, logits_to_spans, uniform_thresholds
from edgeveil.detector.windows import Window, encode_windows
from edgeveil.spans import Span

INFERENCE_BATCH_SIZE = 32


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.thresholds = thresholds or uniform_thresholds()

    def padded_batch(self, windows: Sequence[Window]) -> dict[str, torch.Tensor]:
        width = max(len(window.input_ids) for window in windows)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = [
            window.input_ids + [pad_id] * (width - len(window.input_ids))
            for window in windows
        ]
        mask = [
            [1] * len(window.input_ids) + [0] * (width - len(window.input_ids))
            for window in windows
        ]
        return {
            "input_ids": torch.tensor(input_ids, device=self.device),
            "attention_mask": torch.tensor(mask, device=self.device),
        }

    def window_logits(self, windows: Sequence[Window]) -> list[NDArray[np.float32]]:
        order = sorted(
            range(len(windows)), key=lambda index: len(windows[index].input_ids)
        )
        logits: list[NDArray[np.float32] | None] = [None] * len(windows)
        with torch.inference_mode():
            for start in range(0, len(order), INFERENCE_BATCH_SIZE):
                batch_indices = order[start : start + INFERENCE_BATCH_SIZE]
                batch = self.padded_batch([windows[index] for index in batch_indices])
                output = self.model(**batch).logits.float().cpu().numpy()
                for row, index in enumerate(batch_indices):
                    logits[index] = output[row, : len(windows[index].input_ids)]
        return [row for row in logits if row is not None]

    def detect_batch(self, texts: Sequence[str]) -> list[list[Span]]:
        per_text = encode_windows(self.tokenizer, texts)
        flat = [window for windows in per_text for window in windows]
        flat_logits = self.window_logits(flat)
        results: list[list[Span]] = []
        cursor = 0
        for windows in per_text:
            text_logits = flat_logits[cursor : cursor + len(windows)]
            results.append(logits_to_spans(text_logits, windows, self.thresholds))
            cursor += len(windows)
        return results

    def detect(self, text: str) -> list[Span]:
        return self.detect_batch([text])[0]
