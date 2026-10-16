import json
import plistlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from edgeveil.bench.environment import environment_record
from edgeveil.detector.builds import BuildRecord
from edgeveil.detector.inference import WindowLogits
from edgeveil.detector.windows import Window

POWER_LOOP_SECONDS = 60
IDLE_SECONDS = 30
PLIST_SEPARATOR = b"\x00"
NANOSECONDS_PER_SECOND = 1_000_000_000
MILLIJOULES_PER_JOULE = 1000
RAILS = ("cpu", "gpu", "ane")
RAIL_KEYS = {rail: f"{rail}_power" for rail in RAILS}
RESULT_DECIMALS = 4
PHASE_MARGIN_SECONDS = 2.0


class PowerLogFormatError(ValueError):
    pass


class NoSamplesInPhaseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Phase:
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class PowerLoop:
    idle: Phase
    inference: Phase
    windows: int
    window_tokens: int


@dataclass(frozen=True, slots=True)
class PowerSample:
    end: float
    seconds: float
    milliwatts: dict[str, float]

    @property
    def start(self) -> float:
        return self.end - self.seconds

    def is_within(self, phase: Phase) -> bool:
        earliest = phase.start + PHASE_MARGIN_SECONDS
        latest = phase.end - PHASE_MARGIN_SECONDS
        return earliest <= self.start and self.end <= latest


def timed_phase(action: Callable[[], int]) -> tuple[Phase, int]:
    start = time.time()
    count = action()
    return Phase(start, time.time()), count


def run_power_loop(
    window_logits: Callable[[Sequence[Window]], list[WindowLogits]],
    window: Window,
    seconds: float = POWER_LOOP_SECONDS,
    idle_seconds: float = IDLE_SECONDS,
) -> PowerLoop:
    def idle() -> int:
        time.sleep(idle_seconds)
        return 0

    def infer() -> int:
        deadline = time.monotonic() + seconds
        count = 0
        while time.monotonic() < deadline:
            window_logits([window])
            count += 1
        return count

    idle_phase, _ = timed_phase(idle)
    inference_phase, count = timed_phase(infer)
    return PowerLoop(idle_phase, inference_phase, count, len(window.input_ids))


def checked_processor(document: Mapping[str, Any]) -> Mapping[str, Any]:
    processor = document.get("processor")
    if not isinstance(processor, Mapping) or "elapsed_ns" not in document:
        raise PowerLogFormatError(
            f"sample lacks processor or elapsed_ns; keys: {sorted(document)}"
        )
    missing = [key for key in RAIL_KEYS.values() if key not in processor]
    if missing:
        raise PowerLogFormatError(
            f"processor lacks {missing}; keys: {sorted(processor)}"
        )
    return processor


def first_sample_end(document: Mapping[str, Any]) -> float:
    stamp = document.get("timestamp")
    if not isinstance(stamp, datetime):
        raise PowerLogFormatError(f"first sample lacks a timestamp: {sorted(document)}")
    return stamp.timestamp()


def samples_from_plists(documents: Sequence[Mapping[str, Any]]) -> list[PowerSample]:
    if not documents:
        return []
    end = first_sample_end(documents[0])
    samples: list[PowerSample] = []
    for index, document in enumerate(documents):
        processor = checked_processor(document)
        seconds = document["elapsed_ns"] / NANOSECONDS_PER_SECOND
        if index:
            end += seconds
        milliwatts = {rail: float(processor[key]) for rail, key in RAIL_KEYS.items()}
        samples.append(PowerSample(end=end, seconds=seconds, milliwatts=milliwatts))
    return samples


def read_powermetrics_plist(path: Path) -> list[PowerSample]:
    chunks = [chunk.strip() for chunk in path.read_bytes().split(PLIST_SEPARATOR)]
    return samples_from_plists(
        [plistlib.loads(chunk, aware_datetime=True) for chunk in chunks if chunk]
    )


def samples_within(samples: Sequence[PowerSample], phase: Phase) -> list[PowerSample]:
    inside = [sample for sample in samples if sample.is_within(phase)]
    if not inside:
        raise NoSamplesInPhaseError(
            f"no power sample falls inside {phase.start:.0f} to {phase.end:.0f}"
        )
    return inside


def mean_milliwatts(samples: Sequence[PowerSample]) -> dict[str, float]:
    total_seconds = sum(sample.seconds for sample in samples)
    return {
        rail: sum(s.milliwatts[rail] * s.seconds for s in samples) / total_seconds
        for rail in RAILS
    }


@dataclass(frozen=True, slots=True)
class PowerReport:
    windows: int
    window_tokens: int
    inference_seconds: float
    windows_per_second: float
    samples: int
    idle_milliwatts: dict[str, float]
    inference_milliwatts: dict[str, float]
    millijoules_per_window: dict[str, float]
    millijoules_per_window_above_idle: dict[str, float]


def per_window(milliwatts: Mapping[str, float], loop: PowerLoop) -> dict[str, float]:
    energy = {
        rail: milliwatts[rail] * loop.inference.seconds / loop.windows for rail in RAILS
    }
    return {**energy, "total": sum(energy.values())}


def energy_report(samples: Sequence[PowerSample], loop: PowerLoop) -> PowerReport:
    idle = mean_milliwatts(samples_within(samples, loop.idle))
    inference_samples = samples_within(samples, loop.inference)
    busy = mean_milliwatts(inference_samples)
    above_idle = {rail: busy[rail] - idle[rail] for rail in RAILS}
    return PowerReport(
        windows=loop.windows,
        window_tokens=loop.window_tokens,
        inference_seconds=loop.inference.seconds,
        windows_per_second=loop.windows / loop.inference.seconds,
        samples=len(inference_samples),
        idle_milliwatts=idle,
        inference_milliwatts=busy,
        millijoules_per_window=per_window(busy, loop),
        millijoules_per_window_above_idle=per_window(above_idle, loop),
    )


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, RESULT_DECIMALS)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def write_power_loop(loop: PowerLoop, build: BuildRecord, path: Path) -> None:
    payload = {
        "precision": build.precision,
        "backend": build.backend,
        "compute_units": build.compute_units,
        "build": asdict(build),
        "loop": asdict(loop),
        "environment": environment_record(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_power_loop(payload: Mapping[str, Any]) -> PowerLoop:
    loop = payload["loop"]
    return PowerLoop(
        idle=Phase(**loop["idle"]),
        inference=Phase(**loop["inference"]),
        windows=loop["windows"],
        window_tokens=loop["window_tokens"],
    )


def write_power_results(
    report: PowerReport,
    loop_payload: Mapping[str, Any],
    log_path: Path,
    samples: Sequence[PowerSample],
    path: Path,
) -> None:
    loop = read_power_loop(loop_payload)
    phases = (loop.idle, loop.inference)
    series = [
        {"end": sample.end, "seconds": sample.seconds, **sample.milliwatts}
        for sample in samples
        if any(sample.is_within(phase) for phase in phases)
    ]
    payload = {
        "precision": loop_payload["precision"],
        "backend": loop_payload["backend"],
        "compute_units": loop_payload["compute_units"],
        "build": loop_payload["build"],
        "measured": (
            "powermetrics cpu_power, gpu_power, and ane_power (mW) while one "
            "window ran in a loop at batch size 1; energy per window is mean "
            "power over the loop times its duration divided by windows run; "
            "above idle subtracts the mean of the idle phase before the loop; "
            "sample times are the first plist timestamp plus accumulated "
            "elapsed_ns, and samples within 2 s of a phase edge are left out"
        ),
        "powermetrics_log": log_path.as_posix(),
        **rounded(asdict(report)),
        "samples_in_phases": rounded(series),
        "environment": loop_payload["environment"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
