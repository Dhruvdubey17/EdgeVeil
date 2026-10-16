from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from edgeveil.cli.data import data_app
from edgeveil.detector.builds import (
    Backend,
    ComputeUnits,
    Precision,
    UnsupportedBuildError,
    check_supported,
)
from edgeveil.detector.presets import ModelPreset
from edgeveil.evaluation.detection import Detection, results_stem
from edgeveil.spans import EntityType

if TYPE_CHECKING:
    from collections.abc import Mapping

    from edgeveil.evaluation.accuracy import AccuracyReport
    from edgeveil.evaluation.bootstrap import Estimate

NOT_IMPLEMENTED_EXIT_CODE = 1
USAGE_EXIT_CODE = 2
BYTES_PER_MEGABYTE = 1_000_000
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_LOGGING_STEPS = 100
DEFAULT_RESULTS_DIR = Path("bench") / "results"
DEFAULT_TEST_SET_PATH = DEFAULT_DATA_DIR / "test" / "test_set.jsonl"
DEFAULT_PROXY_PORT = 8787

app = typer.Typer(
    help="Mask personal data on the device before prompts reach a cloud LLM.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
app.add_typer(data_app, name="data")
error_console = Console(stderr=True)
output_console = Console()


class Upstream(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


def exit_not_implemented(command: str, detail: str = "") -> NoReturn:
    error_console.print(f"edgeveil {command}: not implemented yet{detail}")
    raise typer.Exit(code=NOT_IMPLEMENTED_EXIT_CODE)


def exit_unsupported(command: str, error: UnsupportedBuildError) -> NoReturn:
    error_console.print(f"edgeveil {command}: {error}")
    raise typer.Exit(code=USAGE_EXIT_CODE)


def exit_if_unbuilt(
    command: str, model_dir: Path, backend: Backend, precision: Precision
) -> None:
    from edgeveil.detector.builds import build_path

    path = build_path(model_dir, backend, precision)
    if not path.exists():
        error_console.print(
            f"edgeveil {command}: {path} does not exist; build it with "
            f"edgeveil export --backend {backend.value} --precision {precision.value}"
        )
        raise typer.Exit(code=USAGE_EXIT_CODE)


def checked_build(command: str, backend: Backend, precision: Precision) -> None:
    try:
        check_supported(backend, precision)
    except UnsupportedBuildError as error:
        exit_unsupported(command, error)


@app.command(
    name="export",
    help="Build the trained model for one backend and precision under the models "
    "directory. Quantized ONNX builds start from the fp32 ONNX build.",
)
def export_build(
    precision: Annotated[Precision, typer.Option(help="Precision to build.")],
    backend: Annotated[
        Backend, typer.Option(help="Backend to build for.")
    ] = Backend.ONNXRUNTIME,
    preset: Annotated[
        ModelPreset, typer.Option(help="Trained model to export.")
    ] = ModelPreset.DISTILBERT,
    models_dir: Annotated[
        Path, typer.Option(help="Directory with trained models.")
    ] = DEFAULT_MODELS_DIR,
    data_dir: Annotated[
        Path, typer.Option(help="Directory with prepared data, for calibration.")
    ] = DEFAULT_DATA_DIR,
) -> None:
    checked_build("export", backend, precision)
    model_dir = models_dir / preset.value
    if backend is Backend.ONNXRUNTIME:
        from edgeveil.export.onnx_builds import build_onnx

        path = build_onnx(model_dir, precision, data_dir)
    else:
        exit_not_implemented("export", f" for {backend.value}")
    from edgeveil.detector.builds import build_size_bytes

    megabytes = build_size_bytes(path) / BYTES_PER_MEGABYTE
    output_console.print(f"Saved {path} ({megabytes:.1f} MB).")


@app.command(
    help="Measure token-level agreement of builds with the fp32 torch model on "
    "validation windows that calibration never saw, and save one result per build."
)
def parity(
    backend: Annotated[
        Backend, typer.Option(help="Backend whose builds to compare.")
    ] = Backend.ONNXRUNTIME,
    precision: Annotated[
        list[Precision] | None,
        typer.Option(
            help="Build to compare; repeat for several. Default: every built one."
        ),
    ] = None,
    compute_units: Annotated[
        ComputeUnits,
        typer.Option(help="Core ML compute units; other backends ignore it."),
    ] = ComputeUnits.ALL,
    preset: Annotated[
        ModelPreset, typer.Option(help="Trained model to compare.")
    ] = ModelPreset.DISTILBERT,
    models_dir: Annotated[
        Path, typer.Option(help="Directory with trained models.")
    ] = DEFAULT_MODELS_DIR,
    data_dir: Annotated[
        Path, typer.Option(help="Directory with prepared data.")
    ] = DEFAULT_DATA_DIR,
    results_dir: Annotated[
        Path, typer.Option(help="Directory for result files.")
    ] = DEFAULT_RESULTS_DIR,
) -> None:
    from edgeveil.detector.builds import (
        QUANTIZATION_SCHEMES,
        TORCH_WEIGHTS_NAME,
        build_path,
        file_sha256,
        run_name,
    )
    from edgeveil.evaluation.parity import measure_parity, write_parity_results

    model_dir = models_dir / preset.value
    chosen = precision or [
        item
        for item in QUANTIZATION_SCHEMES[backend]
        if build_path(model_dir, backend, item).exists()
    ]
    for item in chosen:
        checked_build("parity", backend, item)
        exit_if_unbuilt("parity", model_dir, backend, item)
    measured = measure_parity(model_dir, data_dir, backend, chosen, compute_units)
    model_sha256 = file_sha256(model_dir / TORCH_WEIGHTS_NAME)
    for item, result in measured.items():
        name = run_name(preset.value, backend, item, compute_units)
        path = results_dir / "parity" / f"{name}.json"
        write_parity_results(result.report, result.build, model_sha256, path)
        report = result.report
        output_console.print(
            f"{name}: token agreement {report.token_agreement:.4f}, entity tokens "
            f"{report.entity_token_agreement:.4f} of {report.reference_entity_tokens}, "
            f"max logit difference {report.max_abs_logit_difference:.4f}. "
            f"Saved {path}."
        )


@app.command(help="Print the masked text and the detected spans.")
def scan(
    file: Annotated[Path, typer.Argument(help="Text file to scan for personal data.")],
) -> None:
    exit_not_implemented("scan")


@app.command(help="Start the OpenAI-compatible masking proxy.")
def proxy(
    port: Annotated[int, typer.Option(help="Port to listen on.")] = DEFAULT_PROXY_PORT,
    upstream: Annotated[
        Upstream, typer.Option(help="LLM provider to forward requests to.")
    ] = Upstream.OPENAI,
) -> None:
    exit_not_implemented("proxy")


@app.command(
    name="eval", help="Evaluate detection accuracy for one precision and backend."
)
def evaluate(
    precision: Annotated[
        Precision, typer.Option(help="Model precision to evaluate.")
    ] = Precision.FP32,
    backend: Annotated[
        Backend, typer.Option(help="Inference backend to evaluate.")
    ] = Backend.TORCH,
    preset: Annotated[
        ModelPreset, typer.Option(help="Trained model to evaluate.")
    ] = ModelPreset.DISTILBERT,
    test_set: Annotated[
        Path, typer.Option(help="Hand-labeled test set JSONL.")
    ] = DEFAULT_TEST_SET_PATH,
    models_dir: Annotated[
        Path, typer.Option(help="Directory with trained models.")
    ] = DEFAULT_MODELS_DIR,
    results_dir: Annotated[
        Path, typer.Option(help="Directory for result files.")
    ] = DEFAULT_RESULTS_DIR,
    compute_units: Annotated[
        ComputeUnits,
        typer.Option(help="Core ML compute units; other backends ignore it."),
    ] = ComputeUnits.ALL,
    show_errors: Annotated[
        Detection | None,
        typer.Option(
            help="Print every strict error of this detection mode with test-set "
            "text around it. The text goes to the terminal only, never to a file."
        ),
    ] = None,
) -> None:
    checked_build("eval", backend, precision)
    from edgeveil.detector.builds import (
        build_record,
        load_detector,
        reference_thresholds_name,
        run_name,
    )
    from edgeveil.evaluation.accuracy import (
        error_example_lines,
        evaluate_test_set,
        load_thresholds,
        write_accuracy_results,
    )

    model_dir = models_dir / preset.value
    exit_if_unbuilt("eval", model_dir, backend, precision)
    thresholds = results_dir / "thresholds" / reference_thresholds_name(preset.value)
    detector = load_detector(
        model_dir, backend, precision, load_thresholds(thresholds), compute_units
    )
    build = build_record(model_dir, backend, precision, compute_units)
    name = run_name(preset.value, backend, precision, compute_units)
    reports = evaluate_test_set(detector, build, test_set, thresholds)
    for detection, report in reports.items():
        stem = results_stem(detection, preset.value, name)
        path = results_dir / "accuracy" / f"{stem}.json"
        write_accuracy_results(report, path)
        output_console.print(f"Saved {detection.value} results to {path}.")
    output_console.print(accuracy_summary(reports[Detection.MODEL]))
    output_console.print(ablation_summary(reports))
    output_console.print(error_summary(reports[Detection.MODEL]))
    if show_errors is not None:
        for line in error_example_lines(reports[show_errors]):
            output_console.print(line, markup=False, highlight=False)


def interval_text(estimate: "Estimate") -> str:
    return f"{estimate.value:.3f} [{estimate.low:.3f}, {estimate.high:.3f}]"


def accuracy_summary(report: "AccuracyReport") -> Table:
    snippets = report.run.snippets
    table = Table(title=f"Test set accuracy, model only ({snippets} snippets)")
    columns = ("Type", "Support", "Precision", "Recall", "F1")
    for column in (*columns, "Overlap P", "Overlap R", "Char recall"):
        table.add_column(column)
    for name, scored in report.scores.items():
        secondary = report.secondary[name]
        estimates = (
            scored.precision,
            scored.recall,
            scored.f1,
            secondary.overlap_precision,
            secondary.overlap_recall,
            secondary.char_recall,
        )
        table.add_row(
            name, str(scored.support), *(interval_text(item) for item in estimates)
        )
    return table


def ablation_summary(reports: "Mapping[Detection, AccuracyReport]") -> Table:
    table = Table(title="Validator ablation, strict F1")
    table.add_column("Type")
    for detection in reports:
        table.add_column(detection.value)
    for name in reports[Detection.MODEL].scores:
        f1_cells = [
            interval_text(report.scores[name].f1) for report in reports.values()
        ]
        table.add_row(name, *f1_cells)
    return table


def error_summary(report: "AccuracyReport") -> Table:
    from edgeveil.evaluation.errors import KINDS_BY_SIDE, error_counts

    counts = error_counts(report.errors)
    table = Table(title="Strict errors, model only")
    table.add_column("Type")
    pairs = [(side, kind) for side, kinds in KINDS_BY_SIDE.items() for kind in kinds]
    for side, kind in pairs:
        table.add_column(f"{side.value}: {kind.value}")
    for entity in EntityType:
        cells = [str(counts[side][entity][kind]) for side, kind in pairs]
        table.add_row(entity.value, *cells)
    return table


@app.command(help="Fine-tune the detector on the augmented OpenPII data.")
def train(
    preset: Annotated[
        ModelPreset, typer.Option(help="Base model to fine-tune.")
    ] = ModelPreset.DISTILBERT,
    data_dir: Annotated[
        Path, typer.Option(help="Directory with prepared data.")
    ] = DEFAULT_DATA_DIR,
    models_dir: Annotated[
        Path, typer.Option(help="Directory for trained models.")
    ] = DEFAULT_MODELS_DIR,
    max_steps: Annotated[
        int | None, typer.Option(help="Stop after this many steps, for smoke runs.")
    ] = None,
    logging_steps: Annotated[
        int, typer.Option(help="Log the training loss every this many steps.")
    ] = DEFAULT_LOGGING_STEPS,
    resume: Annotated[
        bool,
        typer.Option(help="Continue from the last checkpoint in the models directory."),
    ] = False,
    device_batch_size: Annotated[
        int | None,
        typer.Option(
            help="Windows per device step; accumulation keeps the effective batch."
        ),
    ] = None,
) -> None:
    from edgeveil.detector.training import RunOptions, train_detector

    options = RunOptions(
        output_dir=models_dir / preset.value / "checkpoints",
        max_steps=max_steps,
        logging_steps=logging_steps,
        resume=resume,
        device_batch_size=device_batch_size,
    )
    record = train_detector(data_dir, models_dir, preset, options)
    seconds = record.training_seconds
    output_console.print(
        f"Trained {record.base_model} in {seconds:.0f} s on {record.accelerator}."
    )
    output_console.print(f"Saved to {models_dir / record.preset}.")


@app.command(
    name="tune-thresholds",
    help="Pick per-type thresholds on validation by maximizing F2, and save them.",
)
def tune_thresholds(
    preset: Annotated[
        ModelPreset, typer.Option(help="Trained model to tune.")
    ] = ModelPreset.DISTILBERT,
    data_dir: Annotated[
        Path, typer.Option(help="Directory with prepared data.")
    ] = DEFAULT_DATA_DIR,
    models_dir: Annotated[
        Path, typer.Option(help="Directory with trained models.")
    ] = DEFAULT_MODELS_DIR,
    results_dir: Annotated[
        Path, typer.Option(help="Directory for result files.")
    ] = DEFAULT_RESULTS_DIR,
    limit: Annotated[
        int | None,
        typer.Option(help="Use only the first rows of validation, for smoke runs."),
    ] = None,
) -> None:
    from edgeveil.evaluation.thresholds import tune_thresholds as tune
    from edgeveil.evaluation.thresholds import write_threshold_results

    report = tune(models_dir / preset.value, data_dir, limit)
    path = results_dir / "thresholds" / f"{preset.value}-fp32-torch.json"
    write_threshold_results(report, path)
    for entity, point in report.chosen.items():
        output_console.print(
            f"{entity.value}: threshold {point.threshold:.2f}, "
            f"precision {point.precision:.3f}, recall {point.recall:.3f}, "
            f"F2 {point.f_beta:.3f}"
        )
    output_console.print(f"Saved to {path}.")


@app.command(help="Run benchmarks and regenerate every results table.")
def bench(
    results_dir: Annotated[
        Path, typer.Option(help="Directory with result files.")
    ] = DEFAULT_RESULTS_DIR,
) -> None:
    from edgeveil.bench.tables import render_accuracy_tables

    written = render_accuracy_tables(results_dir)
    for path in written:
        output_console.print(path.read_text())
        output_console.print(f"Wrote {path}.")
    if not written:
        output_console.print(f"No accuracy results under {results_dir}.")
