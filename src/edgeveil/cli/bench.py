import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from edgeveil.detector.builds import COREML_BACKENDS, Backend, ComputeUnits, Precision
from edgeveil.detector.presets import ModelPreset

DEFAULT_DATA_DIR = Path("data")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_RESULTS_DIR = Path("bench") / "results"
COOLDOWN_SECONDS = 60
BENCH_TEXT_ROWS = 50
USAGE_EXIT_CODE = 2

bench_app = typer.Typer(
    help="Run latency and placement benchmarks, or with no command, regenerate "
    "every results table from the files under bench/results.",
    invoke_without_command=True,
)
console = Console()
error_console = Console(stderr=True)

ResultsDir = Annotated[Path, typer.Option(help="Directory with result files.")]
ModelsDir = Annotated[Path, typer.Option(help="Directory with trained models.")]
PresetOption = Annotated[ModelPreset, typer.Option(help="Trained model to benchmark.")]
PrecisionsOption = Annotated[
    list[Precision] | None,
    typer.Option(
        "--precision",
        help="Build to benchmark; repeat for several. Default: every built one.",
    ),
]
ComputeUnitsOption = Annotated[
    list[ComputeUnits] | None,
    typer.Option(
        "--compute-units",
        help="Core ML compute units; repeat for several. Default: every setting.",
    ),
]


@bench_app.callback()
def render_tables(
    context: typer.Context, results_dir: ResultsDir = DEFAULT_RESULTS_DIR
) -> None:
    if context.invoked_subcommand is not None:
        return
    from edgeveil.bench.tables import render_accuracy_tables

    written = render_accuracy_tables(results_dir)
    for path in written:
        console.print(path.read_text())
        console.print(f"Wrote {path}.")
    if not written:
        console.print(f"No accuracy results under {results_dir}.")


def built_precisions(
    model_dir: Path, backend: Backend, requested: list[Precision] | None
) -> list[Precision]:
    from edgeveil.detector.builds import (
        QUANTIZATION_SCHEMES,
        UnsupportedBuildError,
        build_path,
        check_supported,
    )

    chosen = requested or list(QUANTIZATION_SCHEMES[backend])
    try:
        for precision in chosen:
            check_supported(backend, precision)
    except UnsupportedBuildError as error:
        error_console.print(f"edgeveil bench: {error}")
        raise typer.Exit(code=USAGE_EXIT_CODE) from error
    built = [p for p in chosen if build_path(model_dir, backend, p).exists()]
    if requested and built != requested:
        error_console.print("edgeveil bench: build the requested precisions first")
        raise typer.Exit(code=USAGE_EXIT_CODE)
    return built


def unit_settings(
    backend: Backend, requested: list[ComputeUnits] | None
) -> list[ComputeUnits | None]:
    if backend not in COREML_BACKENDS:
        return [None]
    return [*(requested or ComputeUnits)]


@bench_app.command(
    help="Time one model call per window at 128, 256, and 512 tokens for each "
    "build, after warmup, and save one result per build and compute-unit setting. "
    "Close other apps first: memory pressure and background load distort timings."
)
def latency(
    backend: Annotated[
        Backend, typer.Option(help="Backend to benchmark.")
    ] = Backend.ONNXRUNTIME,
    precision: PrecisionsOption = None,
    compute_units: ComputeUnitsOption = None,
    preset: PresetOption = ModelPreset.DISTILBERT,
    models_dir: ModelsDir = DEFAULT_MODELS_DIR,
    data_dir: Annotated[
        Path, typer.Option(help="Directory with prepared data, for bench text.")
    ] = DEFAULT_DATA_DIR,
    results_dir: ResultsDir = DEFAULT_RESULTS_DIR,
    cooldown_seconds: Annotated[
        int, typer.Option(help="Idle time before each build is timed.")
    ] = COOLDOWN_SECONDS,
) -> None:
    from edgeveil.bench.latency import (
        LatencyRun,
        bench_windows,
        measure_latency,
        runtime_details,
        system_load,
        write_latency_results,
    )
    from edgeveil.detector.builds import build_record, load_detector, run_name
    from edgeveil.detector.windows import load_tokenizer
    from edgeveil.evaluation.thresholds import load_tuning_rows

    model_dir = models_dir / preset.value
    texts = [row.text for row in load_tuning_rows(data_dir, BENCH_TEXT_ROWS)]
    windows = bench_windows(load_tokenizer(model_dir), texts)
    for item in built_precisions(model_dir, backend, precision):
        for units in unit_settings(backend, compute_units):
            settings = units or ComputeUnits.ALL
            detector = load_detector(model_dir, backend, item, compute_units=settings)
            time.sleep(cooldown_seconds)
            before = system_load()
            timings = measure_latency(detector.window_logits, windows)
            run = LatencyRun(
                build=build_record(model_dir, backend, item, units),
                timings=timings,
                load_before=before,
                load_after=system_load(),
                runtime=runtime_details(detector),
            )
            name = run_name(preset.value, backend, item, units)
            path = results_dir / "latency" / f"{name}.json"
            write_latency_results(run, path)
            summary = ", ".join(
                f"{length}: {timing.p50_ms:.2f} / {timing.p95_ms:.2f} ms"
                for length, timing in timings.items()
            )
            console.print(f"{name} p50 / p95: {summary}. Saved {path}.")


@bench_app.command(
    help="Count the share of Core ML operations whose preferred device is the "
    "Neural Engine, from MLComputePlan, and save one result per build and setting."
)
def placement(
    backend: Annotated[
        Backend, typer.Option(help="Core ML backend whose builds to inspect.")
    ] = Backend.COREML,
    precision: PrecisionsOption = None,
    compute_units: ComputeUnitsOption = None,
    preset: PresetOption = ModelPreset.DISTILBERT,
    models_dir: ModelsDir = DEFAULT_MODELS_DIR,
    results_dir: ResultsDir = DEFAULT_RESULTS_DIR,
) -> None:
    from edgeveil.bench.placement import measure_placement, write_placement_results
    from edgeveil.detector.builds import build_path, build_record, run_name

    model_dir = models_dir / preset.value
    for item in built_precisions(model_dir, backend, precision):
        for units in compute_units or list(ComputeUnits):
            package = build_path(model_dir, backend, item)
            report = measure_placement(package, units)
            name = run_name(preset.value, backend, item, units)
            path = results_dir / "placement" / f"{name}.json"
            build = build_record(model_dir, backend, item, units)
            write_placement_results(report, build, path)
            console.print(
                f"{name}: {report.neural_engine_share:.1%} of {report.operations} "
                f"operations prefer the Neural Engine. Saved {path}."
            )
