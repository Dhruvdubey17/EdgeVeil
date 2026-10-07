from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st
from transformers import PreTrainedTokenizerFast

from edgeveil.detector.alignment import SPECIAL_TOKEN_OFFSET
from edgeveil.detector.windows import (
    MAX_WINDOW_TOKENS,
    WINDOW_OVERLAP_TOKENS,
    encode_windows,
    special_token_frame,
    trim_offset_whitespace,
    window_starts,
)


@given(
    token_count=st.integers(min_value=0, max_value=3000),
    capacity=st.integers(min_value=2, max_value=600),
    overlap_share=st.floats(min_value=0, max_value=0.9),
)
def test_windows_cover_every_token(
    token_count: int, capacity: int, overlap_share: float
) -> None:
    overlap = int(capacity * overlap_share)
    starts = window_starts(token_count, capacity, overlap)
    covered = {
        index
        for start in starts
        for index in range(start, min(start + capacity, token_count))
    }
    assert covered == set(range(token_count))
    assert starts[0] == 0
    assert all(
        later - earlier == capacity - overlap for earlier, later in pairwise(starts)
    )


def test_defaults_match_the_spec() -> None:
    assert MAX_WINDOW_TOKENS == 512
    assert 0 < WINDOW_OVERLAP_TOKENS < MAX_WINDOW_TOKENS


def test_special_token_frame_is_learned_from_the_tokenizer(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    prefix, suffix = special_token_frame(tiny_tokenizer)
    assert tiny_tokenizer.convert_ids_to_tokens(prefix + suffix) == ["[CLS]", "[SEP]"]


def test_long_text_is_split_into_framed_overlapping_windows(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    words = [f"w{index}" for index in range(30)]
    text = " ".join(words)
    windows = encode_windows(tiny_tokenizer, [text], max_tokens=8, overlap=2)[0]
    assert len(windows) > 2
    seen: set[str] = set()
    for window in windows:
        tokens = tiny_tokenizer.convert_ids_to_tokens(window.input_ids)
        assert tokens[0] == "[CLS]"
        assert tokens[-1] == "[SEP]"
        assert len(window.input_ids) <= 8
        assert window.offsets[0] == SPECIAL_TOKEN_OFFSET
        assert window.offsets[-1] == SPECIAL_TOKEN_OFFSET
        for token, (start, end) in zip(tokens[1:-1], window.offsets[1:-1], strict=True):
            assert text[start:end] == token
            seen.add(token)
    assert seen == set(words)


def test_empty_text_yields_one_framed_window(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    [window] = encode_windows(tiny_tokenizer, [""])[0]
    assert tiny_tokenizer.convert_ids_to_tokens(window.input_ids) == ["[CLS]", "[SEP]"]


def test_offsets_never_start_on_whitespace() -> None:
    assert trim_offset_whitespace(" Ada", (0, 4)) == (1, 4)
    assert trim_offset_whitespace("Ada ", (0, 4)) == (0, 3)
    assert trim_offset_whitespace("\N{NO-BREAK SPACE}Ada", (0, 4)) == (1, 4)
    start, end = trim_offset_whitespace("   ", (0, 3))
    assert start == end
