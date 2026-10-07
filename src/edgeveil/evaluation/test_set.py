from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from edgeveil.spans import EntityType, GoldSpan, InvalidSpanError


class InvalidTestSetError(ValueError):
    pass


class SpanRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    start: int
    end: int
    label: EntityType


class SnippetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    id: int
    text: str
    spans: list[SpanRecord]


@dataclass(frozen=True, slots=True)
class LabeledSnippet:
    snippet_id: int
    text: str = field(repr=False)
    spans: tuple[GoldSpan, ...]


def span_problem(text: str, spans: Sequence[GoldSpan]) -> str | None:
    if any(span.end > len(text) for span in spans):
        return "a span ends past the end of the text"
    if any(left.end > right.start for left, right in pairwise(spans)):
        return "spans overlap"
    return None


def snippet_from_record(record: SnippetRecord) -> LabeledSnippet:
    spans = tuple(
        sorted(
            (GoldSpan(span.start, span.end, span.label) for span in record.spans),
            key=lambda span: span.start,
        )
    )
    problem = span_problem(record.text, spans)
    if problem is not None:
        raise InvalidTestSetError(f"snippet {record.id}: {problem}")
    return LabeledSnippet(snippet_id=record.id, text=record.text, spans=spans)


def parse_line(line_number: int, line: str) -> LabeledSnippet:
    try:
        return snippet_from_record(SnippetRecord.model_validate_json(line))
    except (ValidationError, InvalidSpanError) as error:
        raise InvalidTestSetError(
            f"line {line_number}: invalid snippet record"
        ) from error


def load_test_set(path: Path) -> list[LabeledSnippet]:
    lines = path.read_text(encoding="utf-8").splitlines()
    snippets = [
        parse_line(number, line) for number, line in enumerate(lines, 1) if line
    ]
    ids = [snippet.snippet_id for snippet in snippets]
    if len(ids) != len(set(ids)):
        raise InvalidTestSetError("snippet ids are not unique")
    return snippets


def snippet_record(snippet: LabeledSnippet) -> SnippetRecord:
    return SnippetRecord(
        id=snippet.snippet_id,
        text=snippet.text,
        spans=[
            SpanRecord(start=s.start, end=s.end, label=s.label) for s in snippet.spans
        ],
    )


def write_test_set(path: Path, snippets: Iterable[LabeledSnippet]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [snippet_record(snippet).model_dump_json() for snippet in snippets]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
