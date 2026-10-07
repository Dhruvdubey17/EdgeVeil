import json
from pathlib import Path
from typing import Any

import pytest

from edgeveil.evaluation.label_studio import (
    LabelingIssue,
    convert_export,
    import_label_studio_export,
    utf16_to_code_point_offsets,
)
from edgeveil.evaluation.test_set import load_test_set
from edgeveil.spans import EntityType, GoldSpan

TEXT = "Mail ada@example.com or call 555-0100."


def region(start: int, end: int, label: str, text: str | None = None) -> dict[str, Any]:
    return {
        "type": "labels",
        "from_name": "label",
        "to_name": "text",
        "value": {
            "start": start,
            "end": end,
            "text": text or TEXT[start:end],
            "labels": [label],
        },
    }


def task(
    task_id: int, regions: list[dict[str, Any]], text: str = TEXT, **annotation: Any
) -> dict[str, Any]:
    return {
        "id": task_id,
        "data": {"text": text},
        "annotations": [{"was_cancelled": False, "result": regions, **annotation}],
    }


def test_converts_regions_to_sorted_gold_spans() -> None:
    export = [task(1, [region(29, 37, "PHONE"), region(5, 20, "EMAIL")])]
    snippets, issues = convert_export(export)
    assert issues == []
    assert snippets[0].spans == (
        GoldSpan(5, 20, EntityType.EMAIL),
        GoldSpan(29, 37, EntityType.PHONE),
    )


def test_cancelled_annotations_are_ignored() -> None:
    export = [task(1, [region(5, 20, "EMAIL")])]
    export[0]["annotations"].append(
        {"was_cancelled": True, "result": [region(0, 4, "PERSON")]}
    )
    snippets, issues = convert_export(export)
    assert issues == []
    assert len(snippets[0].spans) == 1


@pytest.mark.parametrize(
    ("export", "problem"),
    [
        ([{"id": 1, "data": {"text": TEXT}, "annotations": []}], "no annotation"),
        ([task(1, [region(5, 20, "FAVOURITE_COLOUR")])], "unknown label"),
        ([task(1, [region(4, 20, "EMAIL")])], "whitespace"),
        ([task(1, [region(6, 20, "EMAIL", text="different")])], "does not match"),
        ([task(1, [region(5, 20, "EMAIL"), region(10, 25, "PHONE")])], "overlap"),
    ],
)
def test_reports_labeling_problems_by_task_and_offsets(
    export: list[dict[str, Any]], problem: str
) -> None:
    snippets, issues = convert_export(export)
    assert snippets == []
    assert [issue.task_id for issue in issues] == [1]
    assert problem in issues[0].problem


def test_issue_text_never_contains_the_labeled_value() -> None:
    _, issues = convert_export(
        [task(1, [region(6, 20, "EMAIL", text="ada@example.com")])]
    )
    assert all("example.com" not in str(issue) for issue in issues)


def test_two_live_annotations_are_an_issue() -> None:
    export = [task(1, [region(5, 20, "EMAIL")])]
    export[0]["annotations"].append({"was_cancelled": False, "result": []})
    _, issues = convert_export(export)
    assert "annotations" in issues[0].problem


def test_utf16_offsets_from_the_browser_are_converted() -> None:
    text = "\N{GRINNING FACE} Mail ada@example.com"
    utf16_start = 2 + 6
    snippets, issues = convert_export(
        [
            task(
                1,
                [
                    region(
                        utf16_start, utf16_start + 15, "EMAIL", text="ada@example.com"
                    )
                ],
                text=text,
            )
        ]
    )
    assert issues == []
    span = snippets[0].spans[0]
    assert text[span.start : span.end] == "ada@example.com"


def test_utf16_offset_table_maps_astral_characters() -> None:
    table = utf16_to_code_point_offsets("a\N{GRINNING FACE}b")
    assert table[0] == 0
    assert table[1] == 1
    assert table[3] == 2
    assert table[4] == 3


def test_import_writes_a_test_set_the_loader_reads(tmp_path: Path) -> None:
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps([task(7, [region(5, 20, "EMAIL")])]))
    output = tmp_path / "data" / "test" / "test_set.jsonl"

    issues = import_label_studio_export(export_path, output)

    assert issues == []
    [snippet] = load_test_set(output)
    assert snippet.snippet_id == 7
    assert snippet.text == TEXT
    assert snippet.spans == (GoldSpan(5, 20, EntityType.EMAIL),)


def test_import_with_issues_writes_nothing(tmp_path: Path) -> None:
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps([task(7, [region(4, 20, "EMAIL")])]))
    output = tmp_path / "data" / "test" / "test_set.jsonl"
    issues = import_label_studio_export(export_path, output)
    assert issues
    assert not output.exists()


def test_labeling_issue_is_a_plain_value() -> None:
    issue = LabelingIssue(task_id=3, start=1, end=2, problem="overlap")
    assert issue.task_id == 3
