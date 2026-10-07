import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from edgeveil.evaluation.test_set import LabeledSnippet, write_test_set
from edgeveil.spans import EntityType, GoldSpan

LABEL_REGION_TYPE = "labels"
ASTRAL_UTF16_UNITS = 2
ASTRAL_CODE_POINT_START = 0x10000


class LabelStudioExportError(ValueError):
    pass


class RegionValue(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    start: int
    end: int
    text: str = Field(repr=False)
    labels: list[str]


class Region(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    type: str
    value: RegionValue | None = None


class Annotation(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    was_cancelled: bool = False
    result: list[Region]


class TaskData(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    text: str = Field(repr=False)


class Task(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    id: int
    data: TaskData
    annotations: list[Annotation]


EXPORT_ADAPTER = TypeAdapter(list[Task], config=ConfigDict(hide_input_in_errors=True))


@dataclass(frozen=True, slots=True)
class LabelingIssue:
    task_id: int
    start: int | None
    end: int | None
    problem: str


def utf16_to_code_point_offsets(text: str) -> dict[int, int]:
    table: dict[int, int] = {}
    utf16_offset = 0
    for code_point_offset, character in enumerate(text):
        table[utf16_offset] = code_point_offset
        is_astral = ord(character) >= ASTRAL_CODE_POINT_START
        utf16_offset += ASTRAL_UTF16_UNITS if is_astral else 1
    table[utf16_offset] = len(text)
    return table


def resolve_offsets(text: str, value: RegionValue) -> tuple[int, int] | None:
    if text[value.start : value.end] == value.text:
        return value.start, value.end
    table = utf16_to_code_point_offsets(text)
    start, end = table.get(value.start), table.get(value.end)
    if start is None or end is None or text[start:end] != value.text:
        return None
    return start, end


def region_span(
    task_id: int, text: str, value: RegionValue
) -> tuple[GoldSpan | None, LabelingIssue | None]:
    def issue(problem: str) -> LabelingIssue:
        return LabelingIssue(task_id, value.start, value.end, problem)

    if len(value.labels) != 1:
        return None, issue(f"region has {len(value.labels)} labels instead of one")
    if value.labels[0] not in EntityType.__members__:
        return None, issue(f"unknown label {value.labels[0]}")
    offsets = resolve_offsets(text, value)
    if offsets is None:
        return None, issue("region text does not match its offsets")
    start, end = offsets
    covered = text[start:end]
    if covered != covered.strip():
        return None, issue("span has leading or trailing whitespace")
    return GoldSpan(start, end, EntityType(value.labels[0])), None


def live_annotation(task: Task) -> tuple[Annotation | None, LabelingIssue | None]:
    live = [
        annotation for annotation in task.annotations if not annotation.was_cancelled
    ]
    if not live:
        return None, LabelingIssue(task.id, None, None, "task has no annotation")
    if len(live) > 1:
        return None, LabelingIssue(
            task.id, None, None, f"task has {len(live)} live annotations"
        )
    return live[0], None


def overlap_issues(task_id: int, spans: Sequence[GoldSpan]) -> list[LabelingIssue]:
    return [
        LabelingIssue(task_id, right.start, right.end, "spans overlap")
        for left, right in pairwise(spans)
        if left.end > right.start
    ]


def convert_task(task: Task) -> tuple[LabeledSnippet | None, list[LabelingIssue]]:
    annotation, issue = live_annotation(task)
    if annotation is None:
        return None, [issue] if issue else []
    spans: list[GoldSpan] = []
    issues: list[LabelingIssue] = []
    for region in annotation.result:
        if region.type != LABEL_REGION_TYPE or region.value is None:
            continue
        span, region_issue = region_span(task.id, task.data.text, region.value)
        if span is not None:
            spans.append(span)
        if region_issue is not None:
            issues.append(region_issue)
    spans.sort(key=lambda span: span.start)
    issues.extend(overlap_issues(task.id, spans))
    if issues:
        return None, issues
    return LabeledSnippet(task.id, task.data.text, tuple(spans)), []


def parse_export(export: Any) -> list[Task]:
    try:
        return EXPORT_ADAPTER.validate_python(export)
    except ValidationError as error:
        count = error.error_count()
        raise LabelStudioExportError(f"export has {count} schema errors") from error


def convert_export(export: Any) -> tuple[list[LabeledSnippet], list[LabelingIssue]]:
    snippets: list[LabeledSnippet] = []
    issues: list[LabelingIssue] = []
    for task in parse_export(export):
        snippet, task_issues = convert_task(task)
        if snippet is not None:
            snippets.append(snippet)
        issues.extend(task_issues)
    return snippets, issues


def import_label_studio_export(
    export_path: Path, output_path: Path
) -> list[LabelingIssue]:
    export = json.loads(export_path.read_text(encoding="utf-8"))
    snippets, issues = convert_export(export)
    if not issues:
        write_test_set(output_path, snippets)
    return issues
