from pathlib import Path

import pytest
from typer.testing import CliRunner

from edgeveil.cli import NOT_IMPLEMENTED_EXIT_CODE, app

runner = CliRunner()


@pytest.mark.parametrize(
    "arguments",
    [
        ["scan", "notes.txt"],
        ["proxy", "--port", "8787", "--upstream", "openai"],
        ["eval", "--precision", "int8", "--backend", "coreml"],
        ["bench"],
    ],
)
def test_unbuilt_commands_exit_with_not_implemented(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)
    assert result.exit_code == NOT_IMPLEMENTED_EXIT_CODE
    assert f"edgeveil {arguments[0]}: not implemented yet" in result.stderr


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
