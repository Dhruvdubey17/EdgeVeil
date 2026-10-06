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
