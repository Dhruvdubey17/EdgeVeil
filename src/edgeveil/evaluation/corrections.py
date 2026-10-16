from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from edgeveil.evaluation.label_lint import (
    MAX_LINT_PASSES,
    LintFinding,
    LintRule,
    lint_spans,
)
from edgeveil.evaluation.test_set import LabeledSnippet, SpanRecord, span_problem
from edgeveil.spans import GoldSpan, InvalidSpanError

RULE_SEPARATOR = "; "
CONTEXT_CHARACTERS = 40

Change = tuple[frozenset[GoldSpan], tuple[GoldSpan, ...], frozenset[LintRule]]


class CorrectionError(ValueError):
    pass


class CorrectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    snippet_id: int
    old: list[SpanRecord]
    new: list[SpanRecord]
    rule: str


@dataclass(frozen=True, slots=True)
class Correction:
    snippet_id: int
    old: tuple[GoldSpan, ...]
    new: tuple[GoldSpan, ...]
    rule: str


def gold_spans(records: Sequence[SpanRecord]) -> tuple[GoldSpan, ...]:
    return tuple(GoldSpan(record.start, record.end, record.label) for record in records)


def span_records(spans: Sequence[GoldSpan]) -> list[SpanRecord]:
    return [SpanRecord(start=s.start, end=s.end, label=s.label) for s in spans]


def parse_correction(line_number: int, line: str) -> Correction:
    try:
        record = CorrectionRecord.model_validate_json(line)
        return Correction(
            record.snippet_id,
            gold_spans(record.old),
            gold_spans(record.new),
            record.rule,
        )
    except (ValidationError, InvalidSpanError) as error:
        raise CorrectionError(f"corrections line {line_number} is invalid") from error


def load_corrections(path: Path) -> list[Correction]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        parse_correction(number, line) for number, line in enumerate(lines, 1) if line
    ]


def append_corrections(path: Path, corrections: Sequence[Correction]) -> None:
    lines = [
        CorrectionRecord(
            snippet_id=correction.snippet_id,
            old=span_records(correction.old),
            new=span_records(correction.new),
            rule=correction.rule,
        ).model_dump_json()
        for correction in corrections
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(line + "\n" for line in lines)


def describe(span: GoldSpan) -> str:
    return f"{span.label.value} {span.start}-{span.end}"


def corrected_spans(
    snippet: LabeledSnippet, corrections: Sequence[Correction]
) -> tuple[GoldSpan, ...]:
    spans = list(snippet.spans)
    for correction in corrections:
        missing = [span for span in correction.old if span not in spans]
        if missing:
            raise CorrectionError(
                f"correction for snippet {snippet.snippet_id} no longer matches: "
                f"no {describe(missing[0])} span"
            )
        spans = [span for span in spans if span not in correction.old]
        spans += correction.new
    ordered = tuple(sorted(spans, key=lambda span: span.start))
    problem = span_problem(snippet.text, ordered)
    if problem is not None:
        raise CorrectionError(
            f"corrections for snippet {snippet.snippet_id} leave a problem: {problem}"
        )
    return ordered


def apply_corrections(
    snippets: Sequence[LabeledSnippet], corrections: Sequence[Correction]
) -> list[LabeledSnippet]:
    known = {snippet.snippet_id for snippet in snippets}
    unknown = sorted({c.snippet_id for c in corrections} - known)
    if unknown:
        raise CorrectionError(f"corrections name unknown snippet {unknown[0]}")
    return [
        replace(
            snippet,
            spans=corrected_spans(
                snippet, [c for c in corrections if c.snippet_id == snippet.snippet_id]
            ),
        )
        for snippet in snippets
    ]


def reviewed_spans(corrections: Sequence[Correction]) -> dict[int, set[GoldSpan]]:
    reviewed: dict[int, set[GoldSpan]] = {}
    for correction in corrections:
        reviewed.setdefault(correction.snippet_id, set()).update(correction.new)
    return reviewed


@dataclass(slots=True)
class Lineage:
    origins: dict[GoldSpan, frozenset[GoldSpan]]
    rules: dict[GoldSpan, frozenset[LintRule]]
    removed: list[tuple[frozenset[GoldSpan], frozenset[LintRule]]]

    def apply(self, finding: LintFinding) -> None:
        origins = frozenset().union(*(self.origins.pop(s) for s in finding.old))
        rules = frozenset({finding.rule}).union(
            *(self.rules.pop(s) for s in finding.old)
        )
        if not finding.new:
            self.removed.append((origins, rules))
        for span in finding.new:
            self.origins[span] = origins
            self.rules[span] = rules


def changes(lineage: Lineage) -> list[Change]:
    records: list[Change] = [
        (origins, (span,), lineage.rules[span])
        for span, origins in lineage.origins.items()
        if lineage.rules[span]
    ]
    records += [(origins, (), rules) for origins, rules in lineage.removed]
    groups: list[Change] = []
    for record in records:
        merged = record
        for group in [group for group in groups if record[0] & group[0]]:
            groups.remove(group)
            merged = (merged[0] | group[0], group[1] + merged[1], merged[2] | group[2])
        groups.append(merged)
    return groups


def propose_corrections(
    snippet: LabeledSnippet, reviewed: Collection[GoldSpan] = ()
) -> tuple[list[Correction], list[LintFinding]]:
    lineage = Lineage(
        origins={span: frozenset({span}) for span in snippet.spans},
        rules={span: frozenset() for span in snippet.spans},
        removed=[],
    )
    for _ in range(MAX_LINT_PASSES):
        findings = lint_spans(
            snippet.snippet_id, snippet.text, list(lineage.origins), reviewed
        )
        settled = [finding for finding in findings if finding.is_settled]
        if not settled:
            break
        lineage.apply(settled[0])
    proposals = [
        Correction(
            snippet.snippet_id,
            tuple(sorted(origins, key=lambda span: span.start)),
            tuple(sorted(new, key=lambda span: span.start)),
            RULE_SEPARATOR.join(sorted(rule.value for rule in rules)),
        )
        for origins, new, rules in changes(lineage)
        if set(origins) != set(new)
    ]
    unsettled = [
        finding
        for finding in lint_spans(
            snippet.snippet_id, snippet.text, list(lineage.origins), reviewed
        )
        if not finding.is_settled
    ]
    return proposals, unsettled


def marked_context(text: str, spans: Sequence[GoldSpan], start: int, end: int) -> str:
    left = max(0, start - CONTEXT_CHARACTERS)
    right = min(len(text), end + CONTEXT_CHARACTERS)
    pieces: list[str] = []
    cursor = left
    for span in sorted(spans, key=lambda item: item.start):
        if span.start < cursor or span.end > right:
            continue
        pieces += [
            text[cursor : span.start],
            f"[{span.label.value}: {text[span.start : span.end]}]",
        ]
        cursor = span.end
    pieces.append(text[cursor:right])
    return " ".join("".join(pieces).split())


def change_window(spans: Sequence[GoldSpan]) -> tuple[int, int]:
    return min(span.start for span in spans), max(span.end for span in spans)


def correction_lines(snippet: LabeledSnippet, correction: Correction) -> list[str]:
    start, end = change_window(correction.old + correction.new)
    after = [s for s in snippet.spans if s not in correction.old] + list(correction.new)
    return [
        f"snippet {snippet.snippet_id}: {correction.rule}",
        f"  before: {marked_context(snippet.text, snippet.spans, start, end)}",
        f"  after:  {marked_context(snippet.text, after, start, end)}",
    ]


def finding_lines(snippet: LabeledSnippet, finding: LintFinding) -> list[str]:
    start, end = change_window(finding.old + finding.new)
    return [
        f"snippet {snippet.snippet_id}: {finding.rule.value}",
        f"  labeled: {marked_context(snippet.text, snippet.spans, start, end)}",
    ]
