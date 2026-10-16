from collections.abc import Sequence

import numpy as np
import pytest
from transformers import PreTrainedTokenizerFast

from edgeveil.bench.latency import (
    BenchTextTooShortError,
    bench_windows,
    summarized,
    timed_runs,
)
from edgeveil.bench.placement import (
    CPU,
    NEURAL_ENGINE,
    OperationPlacement,
    tally_placements,
)
from edgeveil.detector.coreml_backend import WindowTooLongError, enumerated_length
from edgeveil.detector.windows import Window

LONG_TEXT = " ".join(f"w{index % 40}" for index in range(600))


def test_bench_windows_have_exactly_the_requested_token_counts(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    windows = bench_windows(tiny_tokenizer.backend_tokenizer, [LONG_TEXT], (16, 64))
    assert {length: len(window.input_ids) for length, window in windows.items()} == {
        16: 16,
        64: 64,
    }


def test_short_bench_text_is_an_error(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    with pytest.raises(BenchTextTooShortError):
        bench_windows(tiny_tokenizer.backend_tokenizer, ["w1 w2"], (16,))


def test_warmup_calls_are_not_timed() -> None:
    calls: list[int] = []

    def window_logits(windows: Sequence[Window]) -> list[np.ndarray]:
        calls.append(len(windows))
        return []

    window = Window(input_ids=[1, 2], offsets=[(0, 0), (0, 0)])
    timings = timed_runs(window_logits, window, warmup=3, runs=5)
    assert len(timings) == 5
    assert calls == [1] * 8


def test_percentiles_interpolate_linearly() -> None:
    timing = summarized(128, [float(value) for value in range(1, 101)])
    assert timing.p50_ms == pytest.approx(50.5)
    assert timing.p95_ms == pytest.approx(95.05)
    assert (timing.min_ms, timing.max_ms) == (1.0, 100.0)


def test_placement_shares_leave_out_unassigned_operations() -> None:
    placements = [
        OperationPlacement("linear", NEURAL_ENGINE, 3.0),
        OperationPlacement("linear", NEURAL_ENGINE, 3.0),
        OperationPlacement("gather", CPU, 2.0),
        OperationPlacement("const", None, 0.0),
    ]
    report = tally_placements(placements)
    assert report.operations == 3
    assert report.unassigned_operations == 1
    assert report.neural_engine_share == pytest.approx(2 / 3)
    assert report.neural_engine_cost_share == pytest.approx(6 / 8)
    assert report.operators_off_neural_engine == {CPU: {"gather": 1}}


@pytest.mark.parametrize(
    ("tokens", "length"), [(2, 128), (128, 128), (129, 256), (512, 512)]
)
def test_windows_pad_to_the_smallest_enumerated_length(
    tokens: int, length: int
) -> None:
    assert enumerated_length(tokens) == length


def test_windows_longer_than_the_largest_shape_are_rejected() -> None:
    with pytest.raises(WindowTooLongError):
        enumerated_length(513)
