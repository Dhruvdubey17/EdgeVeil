from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

from edgeveil.detector.windows import Window
from edgeveil.spans import Span


class Detector(Protocol):
    def detect(self, text: str) -> list[Span]: ...


class BatchDetector(Detector, Protocol):
    tokenizer: Tokenizer

    def window_logits(self, windows: Sequence[Window]) -> list[NDArray[np.float32]]: ...

    def detect_batch(self, texts: Sequence[str]) -> list[list[Span]]: ...
