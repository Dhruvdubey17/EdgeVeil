import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from edgeveil.dataset.openpii import alternation, is_address_gap, is_person_gap
from edgeveil.evaluation.test_set import LabeledSnippet
from edgeveil.spans import EntityType, GoldSpan

MIN_CARD_DIGITS = 13
MAX_LINT_PASSES = 20
TITLES = ("Dr", "Mr", "Mrs", "Ms", "Mx", "Miss", "Prof", "Sir", "Dame", "Rev", "Fr")
SCHEME_WORDS = ("Basic", "Bearer", "Digest")
ID_TYPE_WORDS = (
    "Passport No.",
    "Passport No",
    "Passport",
    "License No.",
    "Licence No.",
    "CDL",
    "DL",
    "ITIN",
    "SSN",
    "SIN",
    "TIN",
    "EIN",
    "NI",
    "ID",
    "No.",
)
LABEL_WORDS = (
    "postal code",
    "First Name",
    "Last Name",
    "ZIP code",
    "postcode",
    "password",
    "Surname",
    "api_key",
    "passwd",
    "mailto",
    "apiKey",
    "email",
    "Phone",
    "token",
    "Name",
    "ZIP",
    "tel",
)
WORD_SEPARATOR = r"(?:\s*[:=#]\s*|\s+)"
TITLE_PREFIX = re.compile(rf"(?:{alternation(TITLES)})\.?\s+")
POSSESSIVE_SUFFIX = re.compile(r"['\u2019]s$")
REGION_SUFFIX = re.compile(r"[\s,]+[A-Z]{2}$")
MASKED_SSN = re.compile(r"[*Xx#\u2022]{3}[- ]?[*Xx#\u2022]{2}[- ]?(\d{4})(?!\d)")
QUOTES = "\"'`"
PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">", "\u201c": "\u201d", "\u2018": "\u2019"}


def leading_words(words: Sequence[str], flags: int = 0) -> re.Pattern[str]:
    return re.compile(rf"(?:{alternation(words)}){WORD_SEPARATOR}", flags)


def trailing_words(words: Sequence[str], flags: int = 0) -> re.Pattern[str]:
    return re.compile(rf"{WORD_SEPARATOR}(?:{alternation(words)})$", flags)


LEADING_WORDS = (leading_words(SCHEME_WORDS), leading_words(LABEL_WORDS, re.I))
TRAILING_WORDS = (trailing_words(SCHEME_WORDS), trailing_words(LABEL_WORDS, re.I))
LEADING_ID_TYPE = leading_words(ID_TYPE_WORDS)
TRAILING_ID_TYPE = trailing_words(ID_TYPE_WORDS)


class LintRule(StrEnum):
    EMAIL_HAS_AT = "EMAIL spans contain @"
    PERSON_TITLE = "titles stay outside PERSON"
    PERSON_COMMA = "a comma splits PERSON"
    PERSON_POSSESSIVE = "a possessive suffix stays outside PERSON"
    PERSON_JOIN = "name parts with only whitespace or connectors between are one PERSON"
    ADDRESS_JOIN = "address parts with an allowed gap between are one ADDRESS"
    ADDRESS_REGION = "a trailing region code stays outside ADDRESS"
    ADDRESS_ANCHOR = "an ADDRESS has a street, building number, or ZIP code"
    EDGE_WORD = "scheme, ID-type, and label words stay outside spans"
    WRAPPING = "punctuation wrapping a value stays outside the span"
    PARTIAL_CARD = "partial card numbers are not labeled"
    PARTIAL_SSN = "partial SSNs are labeled SSN"


@dataclass(frozen=True, slots=True)
class LintFinding:
    snippet_id: int
    rule: LintRule
    old: tuple[GoldSpan, ...]
    new: tuple[GoldSpan, ...]
    is_settled: bool


Check = Callable[[int, str, Sequence[GoldSpan]], list[LintFinding]]


def trimmed(text: str, label: EntityType, start: int, end: int) -> GoldSpan | None:
    value = text[start:end]
    start += len(value) - len(value.lstrip())
    end -= len(value) - len(value.rstrip())
    return GoldSpan(start, end, label) if end > start else None


def replacement(
    snippet_id: int,
    rule: LintRule,
    old: GoldSpan,
    new: Sequence[GoldSpan | None],
) -> LintFinding:
    kept = tuple(span for span in new if span is not None)
    return LintFinding(snippet_id, rule, (old,), kept, is_settled=True)


def value_of(text: str, span: GoldSpan) -> str:
    return text[span.start : span.end]


def email_without_at(
    snippet_id: int, text: str, spans: Sequence[GoldSpan]
) -> list[LintFinding]:
    return [
        LintFinding(snippet_id, LintRule.EMAIL_HAS_AT, (span,), (), is_settled=False)
        for span in spans
        if span.label is EntityType.EMAIL and "@" not in value_of(text, span)
    ]


def person_title(
    snippet_id: int, text: str, spans: Sequence[GoldSpan]
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for span in spans:
        match = TITLE_PREFIX.match(value_of(text, span))
        if span.label is EntityType.PERSON and match:
            new = trimmed(text, span.label, span.start + match.end(), span.end)
            findings.append(replacement(snippet_id, LintRule.PERSON_TITLE, span, [new]))
    return findings


def comma_parts(text: str, span: GoldSpan) -> list[GoldSpan | None]:
    parts: list[GoldSpan | None] = []
    cursor = span.start
    for index in range(span.start, span.end):
        if text[index] == ",":
            parts.append(trimmed(text, span.label, cursor, index))
            cursor = index + 1
    parts.append(trimmed(text, span.label, cursor, span.end))
    return parts


def person_comma(
    snippet_id: int, text: str, spans: Sequence[GoldSpan]
) -> list[LintFinding]:
    return [
        replacement(snippet_id, LintRule.PERSON_COMMA, span, comma_parts(text, span))
        for span in spans
        if span.label is EntityType.PERSON and "," in value_of(text, span)
    ]


def person_possessive(
    snippet_id: int, text: str, spans: Sequence[GoldSpan]
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for span in spans:
        match = POSSESSIVE_SUFFIX.search(value_of(text, span))
        if span.label is EntityType.PERSON and match:
            new = trimmed(text, span.label, span.start, span.start + match.start())
            findings.append(
                replacement(snippet_id, LintRule.PERSON_POSSESSIVE, span, [new])
            )
    return findings


def address_region(
    snippet_id: int, text: str, spans: Sequence[GoldSpan]
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for span in spans:
        match = REGION_SUFFIX.search(value_of(text, span))
        if span.label is EntityType.ADDRESS and match:
            new = trimmed(text, span.label, span.start, span.start + match.start())
            findings.append(
                replacement(snippet_id, LintRule.ADDRESS_REGION, span, [new])
            )
    return findings


def address_without_digits(
    snippet_id: int, text: str, spans: Sequence[GoldSpan]
) -> list[LintFinding]:
    return [
        LintFinding(snippet_id, LintRule.ADDRESS_ANCHOR, (span,), (), is_settled=False)
        for span in spans
        if span.label is EntityType.ADDRESS
        and not any(character.isdigit() for character in value_of(text, span))
    ]


def joined_neighbours(
    snippet_id: int, text: str, spans: Sequence[GoldSpan]
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for left, right in pairwise(spans):
        if left.label is not right.label:
            continue
        gap = text[left.end : right.start]
        joined = (GoldSpan(left.start, right.end, left.label),)
        if left.label is EntityType.PERSON and is_person_gap(gap):
            is_settled = "\n" not in gap
            findings.append(
                LintFinding(
                    snippet_id, LintRule.PERSON_JOIN, (left, right), joined, is_settled
                )
            )
        if left.label is EntityType.ADDRESS and is_address_gap(gap):
            findings.append(
                LintFinding(
                    snippet_id, LintRule.ADDRESS_JOIN, (left, right), joined, True
                )
            )
    return findings


def edge_word_bounds(label: EntityType, value: str) -> tuple[int, int]:
    leading = list(LEADING_WORDS)
    trailing = list(TRAILING_WORDS)
    if label is not EntityType.ADDRESS:
        leading.append(LEADING_ID_TYPE)
        trailing.append(TRAILING_ID_TYPE)
    start, end = 0, len(value)
    for pattern in leading:
        match = pattern.match(value)
        if match and match.end() < len(value):
            start = max(start, match.end())
    for pattern in trailing:
        match = pattern.search(value)
        if match and match.start() > 0:
            end = min(end, match.start())
    return start, end


def edge_words(
    snippet_id: int, text: str, spans: Sequence[GoldSpan]
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for span in spans:
        start, end = edge_word_bounds(span.label, value_of(text, span))
        if (start, end) != (0, span.end - span.start):
            new = trimmed(text, span.label, span.start + start, span.start + end)
            findings.append(replacement(snippet_id, LintRule.EDGE_WORD, span, [new]))
    return findings


def closing_index(value: str) -> int | None:
    opener, closer = value[0], PAIRS[value[0]]
    depth = 0
    for index, character in enumerate(value):
        if character == opener:
            depth += 1
        elif character == closer:
            depth -= 1
            if depth == 0:
                return index
    return None


def wrapping_bounds(value: str) -> tuple[int, int]:
    start, end = 0, len(value)
    if value[0] in QUOTES:
        start = 1
    if value[-1] in QUOTES:
        end = len(value) - 1
    if value[0] in PAIRS:
        closing = closing_index(value)
        if closing is None:
            start = 1
        elif closing == len(value) - 1:
            start, end = 1, len(value) - 1
    closers = {closer: opener for opener, closer in PAIRS.items()}
    if value[-1] in closers and closers[value[-1]] not in value:
        end = len(value) - 1
    return start, end


def wrapping(
    snippet_id: int, text: str, spans: Sequence[GoldSpan]
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for span in spans:
        start, end = wrapping_bounds(value_of(text, span))
        if (start, end) != (0, span.end - span.start):
            new = trimmed(text, span.label, span.start + start, span.start + end)
            findings.append(replacement(snippet_id, LintRule.WRAPPING, span, [new]))
    return findings


def partial_card(
    snippet_id: int, text: str, spans: Sequence[GoldSpan]
) -> list[LintFinding]:
    return [
        replacement(snippet_id, LintRule.PARTIAL_CARD, span, [])
        for span in spans
        if span.label is EntityType.CREDIT_CARD
        and sum(character.isdigit() for character in value_of(text, span))
        < MIN_CARD_DIGITS
    ]


def partial_ssn(
    snippet_id: int, text: str, spans: Sequence[GoldSpan]
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for match in MASKED_SSN.finditer(text):
        digits = GoldSpan(match.start(1), match.end(1), EntityType.SSN)
        if digits in spans:
            continue
        touching = tuple(
            span
            for span in spans
            if span.start < digits.end and digits.start < span.end
        )
        findings.append(
            LintFinding(snippet_id, LintRule.PARTIAL_SSN, touching, (digits,), True)
        )
    return findings


CHECKS: tuple[Check, ...] = (
    email_without_at,
    partial_card,
    partial_ssn,
    wrapping,
    edge_words,
    person_title,
    person_possessive,
    person_comma,
    address_region,
    address_without_digits,
    joined_neighbours,
)


def lint_spans(
    snippet_id: int,
    text: str,
    spans: Sequence[GoldSpan],
    reviewed: Collection[GoldSpan] = (),
) -> list[LintFinding]:
    ordered = sorted(spans, key=lambda span: span.start)
    findings = [
        finding for check in CHECKS for finding in check(snippet_id, text, ordered)
    ]
    return [
        finding
        for finding in findings
        if not finding.old or not all(span in reviewed for span in finding.old)
    ]


def lint_snippet(
    snippet: LabeledSnippet, reviewed: Collection[GoldSpan] = ()
) -> list[LintFinding]:
    return lint_spans(snippet.snippet_id, snippet.text, snippet.spans, reviewed)
