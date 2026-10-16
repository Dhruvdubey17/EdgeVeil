import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from edgeveil.bench.environment import environment_record
from edgeveil.detector.builds import ANE_SEQUENCE_LENGTHS, BuildRecord
from edgeveil.detector.inference import WindowLogits
from edgeveil.detector.onnx_backend import OnnxRuntimeDetector
from edgeveil.detector.protocol import BatchDetector
from edgeveil.detector.windows import Window, encode_windows

BENCH_SEQUENCE_LENGTHS = ANE_SEQUENCE_LENGTHS
WARMUP_RUNS = 20
TIMED_RUNS = 200
BENCH_BATCH_SIZE = 1
PERCENTILES = (50, 95)
NANOSECONDS_PER_MILLISECOND = 1_000_000
RESULT_DECIMALS = 4
BENCH_TEXT_SEPARATOR = "\n"
SYSTEM_PROBES = {
    "thermal": ["pmset", "-g", "therm"],
    "swap": ["sysctl", "-n", "vm.swapusage"],
}


class BenchTextTooShortError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LengthTiming:
    tokens: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    timings_ms: list[float]


def window_of_length(tokenizer: Tokenizer, text: str, length: int) -> Window:
    [first, *_] = encode_windows(tokenizer, [text], max_tokens=length, overlap=0)[0]
    if len(first.input_ids) != length:
        raise BenchTextTooShortError(
            f"benchmark text yields {len(first.input_ids)} tokens, needs {length}"
        )
    return first


def bench_windows(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    lengths: Sequence[int] = BENCH_SEQUENCE_LENGTHS,
) -> dict[int, Window]:
    text = BENCH_TEXT_SEPARATOR.join(texts)
    return {length: window_of_length(tokenizer, text, length) for length in lengths}


def timed_runs(
    window_logits: Callable[[Sequence[Window]], list[WindowLogits]],
    window: Window,
    warmup: int,
    runs: int,
) -> list[float]:
    for _ in range(warmup):
        window_logits([window])
    timings: list[float] = []
    for _ in range(runs):
        started = time.perf_counter_ns()
        window_logits([window])
        timings.append((time.perf_counter_ns() - started) / NANOSECONDS_PER_MILLISECOND)
    return timings


def summarized(tokens: int, timings: Sequence[float]) -> LengthTiming:
    p50, p95 = np.percentile(timings, PERCENTILES)
    return LengthTiming(
        tokens=tokens,
        p50_ms=float(p50),
        p95_ms=float(p95),
        mean_ms=float(np.mean(timings)),
        min_ms=float(min(timings)),
        max_ms=float(max(timings)),
        timings_ms=list(timings),
    )


def measure_latency(
    window_logits: Callable[[Sequence[Window]], list[WindowLogits]],
    windows: dict[int, Window],
    warmup: int = WARMUP_RUNS,
    runs: int = TIMED_RUNS,
) -> dict[int, LengthTiming]:
    return {
        length: summarized(length, timed_runs(window_logits, window, warmup, runs))
        for length, window in windows.items()
    }


def probe_output(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "unavailable"
    return " ".join(result.stdout.split()) or "unavailable"


def system_load() -> dict[str, Any]:
    one, five, fifteen = os.getloadavg()
    probes = {name: probe_output(command) for name, command in SYSTEM_PROBES.items()}
    return {"load_average": [one, five, fifteen], **probes}


def runtime_details(detector: BatchDetector) -> dict[str, Any]:
    details: dict[str, Any] = {"logical_cpus": os.cpu_count()}
    if isinstance(detector, OnnxRuntimeDetector):
        options = detector.session.get_session_options()
        details |= {
            "providers": detector.session.get_providers(),
            "intra_op_num_threads": options.intra_op_num_threads,
            "inter_op_num_threads": options.inter_op_num_threads,
        }
    return details


def rounded_timing(timing: LengthTiming) -> dict[str, Any]:
    values = asdict(timing)
    values["timings_ms"] = [round(item, RESULT_DECIMALS) for item in timing.timings_ms]
    return {
        key: round(value, RESULT_DECIMALS) if isinstance(value, float) else value
        for key, value in values.items()
    }


@dataclass(frozen=True, slots=True)
class LatencyRun:
    build: BuildRecord
    timings: dict[int, LengthTiming]
    load_before: dict[str, Any]
    load_after: dict[str, Any]
    runtime: dict[str, Any]


def write_latency_results(run: LatencyRun, path: Path) -> None:
    payload = {
        "precision": run.build.precision,
        "backend": run.build.backend,
        "compute_units": run.build.compute_units,
        "build": asdict(run.build),
        "measured": (
            "wall time of one model call on one window at batch size 1, after "
            "warmup; tokenization and decoding excluded"
        ),
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "batch_size": BENCH_BATCH_SIZE,
        "percentile_method": "numpy.percentile, linear interpolation",
        "lengths": {
            str(length): rounded_timing(timing)
            for length, timing in run.timings.items()
        },
        "runtime": run.runtime,
        "system_before": run.load_before,
        "system_after": run.load_after,
        "environment": environment_record(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
