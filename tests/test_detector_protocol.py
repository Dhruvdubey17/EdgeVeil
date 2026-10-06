from edgeveil.detector.protocol import Detector
from edgeveil.spans import EntityType, Span, SpanSource

SYNTHETIC_EMAIL = "user@example.com"


class FixedEmailDetector:
    def detect(self, text: str) -> list[Span]:
        start = text.find(SYNTHETIC_EMAIL)
        if start < 0:
            return []
        return [
            Span(
                start=start,
                end=start + len(SYNTHETIC_EMAIL),
                label=EntityType.EMAIL,
                score=1.0,
                source=SpanSource.MODEL,
            )
        ]


def run_detector(detector: Detector, text: str) -> list[Span]:
    return detector.detect(text)


def test_structural_detector_satisfies_protocol() -> None:
    text = f"Write to {SYNTHETIC_EMAIL} today."
    [span] = run_detector(FixedEmailDetector(), text)
    assert text[span.start : span.end] == SYNTHETIC_EMAIL
    assert span.label is EntityType.EMAIL


def test_detector_returns_no_spans_for_clean_text() -> None:
    assert run_detector(FixedEmailDetector(), "Nothing personal here.") == []
