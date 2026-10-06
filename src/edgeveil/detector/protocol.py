from typing import Protocol

from edgeveil.spans import Span


class Detector(Protocol):
    def detect(self, text: str) -> list[Span]: ...
