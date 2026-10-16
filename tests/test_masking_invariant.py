import re
from itertools import pairwise

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from edgeveil.detector.alignment import SPECIAL_TOKEN_OFFSET, Offset
from edgeveil.detector.decoding import (
    logits_to_spans,
    softmax,
    touches_same_type,
    window_spans,
)
from edgeveil.detector.labels import BIO_LABELS, LABEL_TO_ID
from edgeveil.detector.windows import Window
from edgeveil.spans import EntityType, Span
from edgeveil.validators.combiner import combine_spans
from edgeveil.validators.finders import find_validator_spans

TOKEN = re.compile(r"\w+|[^\w\s]")
ALPHABET = "0123456789abcAZ@.-_ "
FRAGMENTS = (
    "202-555-0143",
    "4111 1111 1111 1111",
    "ada.quill@example.com",
    "Zq7Lm2Xp9Rt4Wv8Ks3Nd6Hb1",
    "Ada Quill",
    "900-12-3456",
)
MAX_FRAGMENTS = 5
THRESHOLD_CHOICES = (0.05, 0.3, 0.5, 0.7, 0.95)
MAX_LOGIT_SCALE = 6.0
TOKEN_TYPES = ("O", "PERSON", "EMAIL", "PHONE", "CREDIT_CARD")


def bio_tags(types: list[str]) -> list[str]:
    tags: list[str] = []
    for index, kind in enumerate(types):
        continues = index > 0 and types[index - 1] == kind
        tags.append("O" if kind == "O" else f"{'I' if continues else 'B'}-{kind}")
    return tags


def window_logits(
    draw: st.DrawFn, item: Window, rng: np.random.Generator, scale: float
) -> np.ndarray:
    content = len(item.offsets) - 2
    types = draw(
        st.lists(st.sampled_from(TOKEN_TYPES), min_size=content, max_size=content)
    )
    logits = rng.normal(size=(len(item.offsets), len(BIO_LABELS)))
    for row, tag in enumerate(bio_tags(types), start=1):
        logits[row, LABEL_TO_ID[tag]] += scale
    return logits


def window(offsets: list[Offset]) -> Window:
    framed = [SPECIAL_TOKEN_OFFSET, *offsets, SPECIAL_TOKEN_OFFSET]
    return Window(input_ids=[0] * len(framed), offsets=framed)


@st.composite
def pipeline_inputs(
    draw: st.DrawFn,
) -> tuple[str, list[Window], list[np.ndarray], dict[EntityType, float]]:
    parts = st.one_of(
        st.sampled_from(FRAGMENTS), st.text(alphabet=ALPHABET, max_size=8)
    )
    text = " ".join(draw(st.lists(parts, min_size=1, max_size=MAX_FRAGMENTS)))
    tokens = [(match.start(), match.end()) for match in TOKEN.finditer(text)]
    assume(tokens)
    first_end = draw(st.integers(1, len(tokens)))
    second_start = draw(st.integers(0, first_end))
    windows = [window(tokens[:first_end]), window(tokens[second_start:])]
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    scale = draw(st.floats(0.5, MAX_LOGIT_SCALE))
    logits = [window_logits(draw, item, rng, scale) for item in windows]
    thresholds = {
        entity: draw(st.sampled_from(THRESHOLD_CHOICES)) for entity in EntityType
    }
    return text, windows, logits, thresholds


def covered(spans: list[Span]) -> set[int]:
    return {index for span in spans for index in range(span.start, span.end)}


@given(pipeline_inputs())
def test_masking_covers_every_model_and_validator_span(
    case: tuple[str, list[Window], list[np.ndarray], dict[EntityType, float]],
) -> None:
    text, windows, logits, thresholds = case
    above_threshold = [
        span
        for window_logits, item in zip(logits, windows, strict=True)
        for span in window_spans(softmax(window_logits), item.offsets, thresholds)
    ]
    validators = find_validator_spans(text)
    merged = logits_to_spans(logits, windows, thresholds)
    masked = combine_spans(text, merged, validators)

    detected = covered(above_threshold) | covered(validators)
    must_mask = {index for index in detected if not text[index].isspace()}
    assert must_mask <= covered(masked)
    assert covered(masked) <= detected
    assert all(not left.overlaps(right) for left, right in pairwise(masked))
    assert not any(touches_same_type(left, right) for left, right in pairwise(merged))
