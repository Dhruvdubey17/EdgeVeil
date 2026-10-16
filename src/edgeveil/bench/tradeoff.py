import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edgeveil.detector.builds import COREML_BACKENDS, Backend, ComputeUnits, Precision
from edgeveil.evaluation.detection import Detection

ACCURACY_DIR_NAME = "accuracy"
PARITY_DIR_NAME = "parity"
LATENCY_DIR_NAME = "latency"
PLACEMENT_DIR_NAME = "placement"
POWER_DIR_NAME = "power"
TRADEOFF_TABLE_NAME = "accuracy-latency.md"
NOT_MEASURED = "not measured"
NOT_APPLICABLE = "n/a"
REFERENCE = "reference"
REFERENCE_BUILD = (Backend.TORCH, Precision.FP32)
OVERALL = "overall"
BYTES_PER_MEGABYTE = 1_000_000
LATENCY_LENGTHS = ("128", "256", "512")
TRADEOFF_COLUMNS = (
    "Backend",
    "Precision",
    "Compute units",
    "Size (MB)",
    "F1, model (95% CI)",
    "F1, model and validators",
    "Token agreement",
    "Entity token agreement",
    *(f"p50 / p95 ms, {length} tokens" for length in LATENCY_LENGTHS),
    "Neural Engine ops",
    "mJ per 512-token window, above idle",
)

BuildKey = tuple[Backend, Precision, ComputeUnits | None]


@dataclass(slots=True)
class BuildResults:
    sources: list[str] = field(default_factory=list)
    accuracy: Mapping[str, Any] | None = None
    combined: Mapping[str, Any] | None = None
    parity: Mapping[str, Any] | None = None
    latency: Mapping[str, Any] | None = None
    placement: Mapping[str, Any] | None = None
    power: Mapping[str, Any] | None = None


def build_key(payload: Mapping[str, Any]) -> BuildKey:
    units = payload.get("compute_units")
    return (
        Backend(payload["backend"]),
        Precision(payload["precision"]),
        None if units is None else ComputeUnits(units),
    )


def sort_key(key: BuildKey) -> tuple[int, int, int]:
    backend, precision, units = key
    units_rank = -1 if units is None else list(ComputeUnits).index(units)
    return (
        list(Backend).index(backend),
        list(Precision).index(precision),
        units_rank,
    )


def result_files(results_dir: Path, directory: str) -> list[Path]:
    return sorted((results_dir / directory).glob("*.json"))


def accuracy_slot(payload: Mapping[str, Any]) -> str | None:
    detection = Detection(payload["detection"])
    if detection is Detection.MODEL:
        return "accuracy"
    if detection is Detection.COMBINED:
        return "combined"
    return None


def collect_results(results_dir: Path) -> dict[BuildKey, BuildResults]:
    collected: dict[BuildKey, BuildResults] = {}
    slots = {
        PARITY_DIR_NAME: "parity",
        LATENCY_DIR_NAME: "latency",
        PLACEMENT_DIR_NAME: "placement",
        POWER_DIR_NAME: "power",
    }
    for directory in (ACCURACY_DIR_NAME, *slots):
        for path in result_files(results_dir, directory):
            payload = json.loads(path.read_text())
            slot = (
                accuracy_slot(payload)
                if directory == ACCURACY_DIR_NAME
                else slots[directory]
            )
            if slot is None:
                continue
            results = collected.setdefault(build_key(payload), BuildResults())
            setattr(results, slot, payload)
            results.sources.append(path.as_posix())
    return collected


def estimate_text(estimate: Mapping[str, float]) -> str:
    return f"{estimate['value']:.3f} [{estimate['low']:.3f}, {estimate['high']:.3f}]"


def f1_cell(payload: Mapping[str, Any] | None, with_interval: bool) -> str:
    if payload is None:
        return NOT_MEASURED
    f1 = payload["scores"][OVERALL]["f1"]
    return estimate_text(f1) if with_interval else f"{f1['value']:.3f}"


def parity_cell(key: BuildKey, payload: Mapping[str, Any] | None, measure: str) -> str:
    if key[:2] == REFERENCE_BUILD:
        return REFERENCE
    if payload is None:
        return NOT_MEASURED
    return f"{payload['parity'][measure]:.4f}"


def latency_cell(payload: Mapping[str, Any] | None, length: str) -> str:
    if payload is None or length not in payload["lengths"]:
        return NOT_MEASURED
    timing = payload["lengths"][length]
    return f"{timing['p50_ms']:.2f} / {timing['p95_ms']:.2f}"


def placement_cell(backend: Backend, payload: Mapping[str, Any] | None) -> str:
    if backend not in COREML_BACKENDS:
        return NOT_APPLICABLE
    if payload is None:
        return NOT_MEASURED
    share = payload["neural_engine_share"]
    return f"{share:.1%} of {payload['operations']}"


def power_cell(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return NOT_MEASURED
    return f"{payload['millijoules_per_window_above_idle']['total']:.1f}"


def size_cell(results: BuildResults) -> str:
    payloads = (
        results.accuracy,
        results.latency,
        results.parity,
        results.placement,
        results.power,
    )
    for payload in payloads:
        if payload is not None and payload.get("build"):
            return f"{payload['build']['size_bytes'] / BYTES_PER_MEGABYTE:.1f}"
    return NOT_MEASURED


def tradeoff_row(key: BuildKey, results: BuildResults) -> list[str]:
    backend, precision, units = key
    return [
        backend.value,
        precision.value,
        NOT_APPLICABLE if units is None else units.value,
        size_cell(results),
        f1_cell(results.accuracy, with_interval=True),
        f1_cell(results.combined, with_interval=False),
        parity_cell(key, results.parity, "token_agreement"),
        parity_cell(key, results.parity, "entity_token_agreement"),
        *(latency_cell(results.latency, length) for length in LATENCY_LENGTHS),
        placement_cell(backend, results.placement),
        power_cell(results.power),
    ]


def markdown_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines)


def latency_machines(collected: Mapping[BuildKey, BuildResults]) -> list[str]:
    machines = {
        f"{results.latency['environment']['chip']}, "
        f"{results.latency['environment']['os']}"
        for results in collected.values()
        if results.latency is not None
    }
    return sorted(machines)


def tradeoff_caption(collected: Mapping[BuildKey, BuildResults]) -> str:
    machines = "; ".join(latency_machines(collected)) or NOT_MEASURED
    return (
        "Accuracy against latency for every build. F1 is strict, overall, on the "
        "dev-test set with the fp32 torch thresholds. Token agreement is with fp32 "
        "torch on 2,000 validation windows. Latency is one window at batch size 1, "
        "model call only, after warmup. Energy is CPU, GPU, and Neural Engine power "
        "from powermetrics during a fixed loop, minus idle, divided by windows run. "
        f"Latency machine: {machines}."
    )


def is_cpu_only(key: BuildKey) -> bool:
    backend, _, units = key
    return backend is Backend.ONNXRUNTIME or units is ComputeUnits.CPU_ONLY


def is_neural_engine_only(key: BuildKey) -> bool:
    return key[2] is ComputeUnits.CPU_AND_NE


def p95_at_longest(results: BuildResults) -> float | None:
    if results.latency is None:
        return None
    timing = results.latency["lengths"].get(LATENCY_LENGTHS[-1])
    return None if timing is None else timing["p95_ms"]


def fastest(
    collected: Mapping[BuildKey, BuildResults], is_member: Callable[[BuildKey], bool]
) -> tuple[BuildKey, float] | None:
    timed = [
        (key, p95)
        for key, results in collected.items()
        if is_member(key) and (p95 := p95_at_longest(results)) is not None
    ]
    return min(timed, key=lambda item: item[1], default=None)


def build_label(key: BuildKey) -> str:
    backend, precision, units = key
    suffix = "" if units is None else f" {units.value}"
    return f"{backend.value} {precision.value}{suffix}"


def baseline_summary(collected: Mapping[BuildKey, BuildResults]) -> str:
    neural_engine = fastest(collected, is_neural_engine_only)
    cpu = fastest(collected, is_cpu_only)
    if neural_engine is None or cpu is None:
        return "Fastest Neural Engine and CPU-only builds: not measured."
    (ne_key, ne_p95), (cpu_key, cpu_p95) = neural_engine, cpu
    return (
        f"Fastest build with compute units cpu-and-ne: {build_label(ne_key)}, "
        f"{ne_p95:.2f} ms p95 at {LATENCY_LENGTHS[-1]} tokens. Fastest CPU-only "
        f"build: {build_label(cpu_key)}, {cpu_p95:.2f} ms, "
        f"{cpu_p95 / ne_p95:.1f}x slower."
    )


def tradeoff_table(collected: Mapping[BuildKey, BuildResults]) -> str:
    keys = sorted(collected, key=sort_key)
    rows = [tradeoff_row(key, collected[key]) for key in keys]
    sources = sorted(source for key in keys for source in collected[key].sources)
    listed = "\n".join(f"- `{source}`" for source in sources)
    return (
        "\n\n".join(
            [
                tradeoff_caption(collected),
                markdown_table(TRADEOFF_COLUMNS, rows),
                baseline_summary(collected),
                f"Sources:\n\n{listed}",
            ]
        )
        + "\n"
    )


def render_tradeoff_tables(results_dir: Path, tables_dir: Path) -> list[Path]:
    collected = collect_results(results_dir)
    if not collected:
        return []
    path = tables_dir / TRADEOFF_TABLE_NAME
    path.write_text(tradeoff_table(collected))
    return [path]
