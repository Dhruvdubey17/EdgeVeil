import json
from pathlib import Path
from typing import Any

from edgeveil.bench.tradeoff import NOT_MEASURED, render_tradeoff_tables

ESTIMATE = {"value": 0.7, "low": 0.65, "high": 0.75}
BUILD = {"size_bytes": 66_200_000}


def accuracy(detection: str, backend: str | None, precision: str | None) -> Any:
    return {
        "detection": detection,
        "preset": "distilbert",
        "backend": backend,
        "precision": precision,
        "compute_units": "all" if backend == "coreml" else None,
        "build": BUILD if backend else None,
        "scores": {"overall": {"f1": ESTIMATE}},
    }


def latency() -> Any:
    timing = {"p50_ms": 4.25, "p95_ms": 5.5}
    return {
        "backend": "onnxruntime",
        "precision": "int8-dynamic",
        "compute_units": None,
        "build": BUILD,
        "lengths": {"128": timing, "256": timing, "512": timing},
        "environment": {"chip": "Apple M3", "os": "macOS-26.1-arm64"},
    }


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def rendered_rows(tmp_path: Path) -> dict[str, list[str]]:
    results = tmp_path / "results"
    tables = tmp_path / "tables"
    tables.mkdir()
    write(
        results / "accuracy" / "a.json",
        accuracy("model", "onnxruntime", "int8-dynamic"),
    )
    write(results / "accuracy" / "b.json", accuracy("model", "coreml", "int4"))
    write(results / "accuracy" / "v.json", accuracy("validators", None, None))
    write(results / "latency" / "c.json", latency())
    [path] = render_tradeoff_tables(results, tables)
    rows = [line for line in path.read_text().splitlines() if line.startswith("| ")]
    return {row.split(" | ")[0].removeprefix("| "): row.split(" | ") for row in rows}


def test_rows_join_accuracy_and_latency_for_the_same_build(tmp_path: Path) -> None:
    rows = rendered_rows(tmp_path)
    onnx = rows["onnxruntime"]
    assert onnx[1:4] == ["int8-dynamic", "n/a", "66.2"]
    assert onnx[4] == "0.700 [0.650, 0.750]"
    assert onnx[8] == "4.25 / 5.50"
    assert onnx[-1] == "n/a |"


def test_missing_measurements_say_not_measured(tmp_path: Path) -> None:
    coreml = rendered_rows(tmp_path)["coreml"]
    assert coreml[1:3] == ["int4", "all"]
    assert coreml[8] == NOT_MEASURED
    assert coreml[-1] == f"{NOT_MEASURED} |"
    assert coreml[6] == NOT_MEASURED
