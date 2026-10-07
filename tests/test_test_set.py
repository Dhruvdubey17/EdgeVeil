import ast
import json
from pathlib import Path

import pytest

from edgeveil.evaluation.test_set import InvalidTestSetError, load_test_set

SOURCE_ROOT = Path(__file__).parent.parent / "src" / "edgeveil"
LOADER_MODULE = "edgeveil.evaluation.test_set"


def write_lines(path: Path, *records: object) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def test_loads_snippets_with_hidden_text_in_repr(tmp_path: Path) -> None:
    path = write_lines(
        tmp_path / "t.jsonl",
        {
            "id": 1,
            "text": "Mail ada@example.com",
            "spans": [{"start": 5, "end": 20, "label": "EMAIL"}],
        },
    )
    [snippet] = load_test_set(path)
    assert "example.com" not in repr(snippet)


@pytest.mark.parametrize(
    "record",
    [
        {"id": 1, "text": "short", "spans": [{"start": 0, "end": 9, "label": "EMAIL"}]},
        {
            "id": 1,
            "text": "abcdef",
            "spans": [
                {"start": 0, "end": 4, "label": "EMAIL"},
                {"start": 2, "end": 5, "label": "PHONE"},
            ],
        },
        {"id": 1, "text": "abcdef", "spans": [{"start": 0, "end": 4, "label": "NOPE"}]},
    ],
)
def test_rejects_invalid_snippets(tmp_path: Path, record: dict[str, object]) -> None:
    with pytest.raises(InvalidTestSetError):
        load_test_set(write_lines(tmp_path / "t.jsonl", record))


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    record = {"id": 1, "text": "abc", "spans": []}
    with pytest.raises(InvalidTestSetError):
        load_test_set(write_lines(tmp_path / "t.jsonl", record, record))


def test_error_messages_hide_snippet_text(tmp_path: Path) -> None:
    record = {
        "id": 1,
        "text": "ada@example.com",
        "spans": [{"start": 0, "end": 99, "label": "EMAIL"}],
    }
    with pytest.raises(InvalidTestSetError) as raised:
        load_test_set(write_lines(tmp_path / "t.jsonl", record))
    assert "example.com" not in str(raised.value)


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def test_only_the_evaluation_package_imports_the_test_set_loader() -> None:
    importers = [
        path.relative_to(SOURCE_ROOT)
        for path in SOURCE_ROOT.rglob("*.py")
        if LOADER_MODULE in imported_modules(path)
    ]
    assert importers
    assert all(path.parts[0] == "evaluation" for path in importers)
