from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from edgeveil.detector.alignment import SPECIAL_TOKEN_OFFSET, Offset

MAX_WINDOW_TOKENS = 512
WINDOW_OVERLAP_TOKENS = 128
FRAME_PROBE_TEXT = "a"
TOKENIZER_FILE_NAME = "tokenizer.json"


@dataclass(frozen=True, slots=True)
class Window:
    input_ids: list[int]
    offsets: list[Offset]


def window_starts(token_count: int, capacity: int, overlap: int) -> list[int]:
    step = capacity - overlap
    starts = [0]
    while starts[-1] + capacity < token_count:
        starts.append(starts[-1] + step)
    return starts


@dataclass(frozen=True, slots=True)
class WindowLayout:
    prefix: list[int]
    suffix: list[int]
    max_tokens: int
    overlap: int

    @property
    def capacity(self) -> int:
        return self.max_tokens - len(self.prefix) - len(self.suffix)


def load_tokenizer(model_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(model_dir / TOKENIZER_FILE_NAME))


def special_token_frame(tokenizer: Tokenizer) -> tuple[list[int], list[int]]:
    framed = tokenizer.encode(FRAME_PROBE_TEXT).ids
    bare = tokenizer.encode(FRAME_PROBE_TEXT, add_special_tokens=False).ids
    for start in range(len(framed) - len(bare) + 1):
        if framed[start : start + len(bare)] == bare:
            return framed[:start], framed[start + len(bare) :]
    raise ValueError("tokenizer adds special tokens inside the text")


def trim_offset_whitespace(text: str, offset: Offset) -> Offset:
    start, end = offset
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end:
        return start, start
    return start, end


def frame_windows(
    text: str, input_ids: Sequence[int], offsets: Sequence[Offset], layout: WindowLayout
) -> list[Window]:
    trimmed = [
        trim_offset_whitespace(text, (offset[0], offset[1])) for offset in offsets
    ]
    prefix_offsets = [SPECIAL_TOKEN_OFFSET] * len(layout.prefix)
    suffix_offsets = [SPECIAL_TOKEN_OFFSET] * len(layout.suffix)
    windows: list[Window] = []
    for start in window_starts(len(input_ids), layout.capacity, layout.overlap):
        end = start + layout.capacity
        windows.append(
            Window(
                input_ids=[*layout.prefix, *input_ids[start:end], *layout.suffix],
                offsets=[*prefix_offsets, *trimmed[start:end], *suffix_offsets],
            )
        )
    return windows


def encode_windows(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    max_tokens: int = MAX_WINDOW_TOKENS,
    overlap: int = WINDOW_OVERLAP_TOKENS,
) -> list[list[Window]]:
    prefix, suffix = special_token_frame(tokenizer)
    layout = WindowLayout(prefix, suffix, max_tokens, overlap)
    encoded = tokenizer.encode_batch(list(texts), add_special_tokens=False)
    return [
        frame_windows(text, encoding.ids, encoding.offsets, layout)
        for text, encoding in zip(texts, encoded, strict=True)
    ]
