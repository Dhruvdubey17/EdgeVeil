from pathlib import Path

import pytest
from typer.testing import CliRunner

from edgeveil.cli import NOT_IMPLEMENTED_EXIT_CODE, USAGE_EXIT_CODE, app

runner = CliRunner()


@pytest.mark.parametrize(
    "arguments",
    [
        ["scan", "notes.txt"],
        ["proxy", "--port", "8787", "--upstream", "openai"],
    ],
)
def test_unbuilt_commands_exit_with_not_implemented(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)
    assert result.exit_code == NOT_IMPLEMENTED_EXIT_CODE
    assert f"edgeveil {arguments[0]}: not implemented yet" in result.stderr


@pytest.mark.parametrize("command", ["eval", "export"])
def test_unsupported_precision_names_the_supported_ones(command: str) -> None:
    arguments = [command, "--precision", "int8", "--backend", "onnxruntime"]
    result = runner.invoke(app, arguments)
    assert result.exit_code == USAGE_EXIT_CODE
    assert "int8-dynamic" in result.stderr


def test_eval_of_a_missing_build_says_how_to_build_it(tmp_path: Path) -> None:
    arguments = ["eval", "--precision", "fp16", "--backend", "onnxruntime"]
    result = runner.invoke(app, [*arguments, "--models-dir", str(tmp_path)])
    assert result.exit_code == USAGE_EXIT_CODE
    assert "edgeveil export --backend onnxruntime --precision fp16" in result.stderr


def test_without_arguments_shows_every_command() -> None:
    result = runner.invoke(app, [])
    for command in ("scan", "proxy", "eval", "bench"):
        assert command in result.output


def test_tracebacks_never_show_local_variables() -> None:
    assert app.pretty_exceptions_show_locals is False


def test_import_labels_reports_a_missing_export_without_a_traceback(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app, ["data", "import-labels", str(tmp_path / "missing.json")]
    )
    assert result.exit_code != 0
    assert "does not exist" in result.output
    assert "Traceback" not in result.output


def test_bench_reports_when_there_are_no_results(tmp_path: Path) -> None:
    result = runner.invoke(app, ["bench", "--results-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No accuracy results" in result.output
