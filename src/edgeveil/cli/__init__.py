from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from edgeveil.cli.data import data_app
from edgeveil.detector.presets import ModelPreset
from edgeveil.evaluation.detection import Detection, results_stem
from edgeveil.spans import EntityType

if TYPE_CHECKING:
    from collections.abc import Mapping

    from edgeveil.evaluation.accuracy import AccuracyReport
    from edgeveil.evaluation.bootstrap import Estimate

NOT_IMPLEMENTED_EXIT_CODE = 1
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


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


class Backend(StrEnum):
    TORCH = "torch"
    ONNXRUNTIME = "onnxruntime"
    COREML = "coreml"
    QUALCOMM = "qualcomm"


def exit_not_implemented(command: str, detail: str = "") -> NoReturn:
    error_console.print(f"edgeveil {command}: not implemented yet{detail}")
    raise typer.Exit(code=NOT_IMPLEMENTED_EXIT_CODE)


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
    show_errors: Annotated[
        Detection | None,
        typer.Option(
            help="Print every strict error of this detection mode with test-set "
            "text around it. The text goes to the terminal only, never to a file."
        ),
    ] = None,
) -> None:
    if (precision, backend) != (Precision.FP32, Backend.TORCH):
        exit_not_implemented("eval", f" for {precision.value} on {backend.value}")
    from edgeveil.evaluation.accuracy import (
        error_example_lines,
        evaluate_test_set,
        write_accuracy_results,
    )

    run_name = f"{preset.value}-{precision.value}-{backend.value}"
    thresholds = results_dir / "thresholds" / f"{run_name}.json"
    reports = evaluate_test_set(models_dir / preset.value, test_set, thresholds)
    for detection, report in reports.items():
        stem = results_stem(detection, preset.value, run_name)
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
