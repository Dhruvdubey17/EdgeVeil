from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

NOT_IMPLEMENTED_EXIT_CODE = 1
DEFAULT_PROXY_PORT = 8787

app = typer.Typer(
    help="Mask personal data on the device before prompts reach a cloud LLM.",
    no_args_is_help=True,
)
error_console = Console(stderr=True)


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


def exit_not_implemented(command: str) -> NoReturn:
    error_console.print(f"edgeveil {command}: not implemented yet")
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
) -> None:
    exit_not_implemented("eval")


@app.command(help="Run benchmarks and regenerate every results table.")
def bench() -> None:
    exit_not_implemented("bench")
