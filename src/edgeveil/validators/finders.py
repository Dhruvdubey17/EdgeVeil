import math
import re
from collections import Counter
from collections.abc import Iterator, Sequence

from edgeveil.spans import EntityType, Span, SpanSource

VALIDATOR_SCORE = 1.0
MIN_CARD_DIGITS = 13
MAX_CARD_DIGITS = 19
LUHN_MODULUS = 10
LUHN_DOUBLED_CEILING = 9
MIN_PHONE_DIGITS = 7
MAX_PHONE_DIGITS = 15
MIN_SECRET_LENGTH = 20
MIN_SECRET_ENTROPY_BITS = 3.5
SECRET_TRAILING_PUNCTUATION = ".-"

DIGIT_GROUPS = re.compile(r"(?<![\w-])\d+(?:[ -]\d+)*(?![\w-])")
EMAIL = re.compile(
    r"(?<![\w.%+-])"
    r"[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*"
    r"@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}"
    r"(?![\w-])"
)
PHONE_SHAPES = (
    r"(?:\+1[ .-]?)?(?:\(\d{3}\)\s?|\d{3}[ .-])\d{3}[ .-]\d{4}",
    r"\d{3}-\d{4}",
    r"\+\d{1,3}(?:[ .-]?\(\d{1,5}\))?(?:[ .-]\d{1,8}){1,5}",
    r"0\d{2,4}[ -]\d{3,4}[ -]?\d{3,4}",
)
PHONE = re.compile(r"(?<![\w.+-])(?:" + "|".join(PHONE_SHAPES) + r")(?![\w-]|\.\d)")
SECRET_CANDIDATE = re.compile(r"(?<![\w+/.-])[A-Za-z0-9_+/.-]{20,}={0,2}(?![\w+/=-])")
HEX_ONLY = re.compile(r"[0-9a-fA-F-]+")


def validator_span(start: int, end: int, label: EntityType) -> Span:
    return Span(start, end, label, VALIDATOR_SCORE, SpanSource.VALIDATOR)


def digits_of(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def passes_luhn(digits: str) -> bool:
    total = 0
    for position, character in enumerate(reversed(digits)):
        value = int(character)
        if position % 2:
            value *= 2
            if value > LUHN_DOUBLED_CEILING:
                value -= LUHN_DOUBLED_CEILING
        total += value
    return total % LUHN_MODULUS == 0


def is_valid_card_number(value: str) -> bool:
    digits = digits_of(value)
    return MIN_CARD_DIGITS <= len(digits) <= MAX_CARD_DIGITS and passes_luhn(digits)


def has_one_separator_style(value: str) -> bool:
    return len({character for character in value if not character.isdigit()}) <= 1


def group_bounds(run: re.Match[str]) -> list[tuple[int, int]]:
    return [
        (run.start() + group.start(), run.start() + group.end())
        for group in re.finditer(r"\d+", run.group())
    ]


def longest_card_in_run(text: str, run: re.Match[str]) -> tuple[int, int] | None:
    groups = group_bounds(run)
    candidates = [
        (groups[first][0], groups[last][1])
        for first in range(len(groups))
        for last in range(first, len(groups))
    ]
    valid = [
        (start, end)
        for start, end in candidates
        if has_one_separator_style(text[start:end])
        and is_valid_card_number(text[start:end])
    ]
    return max(valid, key=lambda bounds: bounds[1] - bounds[0], default=None)


def card_number_spans(text: str) -> Iterator[Span]:
    for run in DIGIT_GROUPS.finditer(text):
        bounds = longest_card_in_run(text, run)
        if bounds is not None:
            yield validator_span(*bounds, EntityType.CREDIT_CARD)


def email_spans(text: str) -> Iterator[Span]:
    for match in EMAIL.finditer(text):
        yield validator_span(match.start(), match.end(), EntityType.EMAIL)


def has_phone_digit_count(value: str) -> bool:
    return MIN_PHONE_DIGITS <= len(digits_of(value)) <= MAX_PHONE_DIGITS


def phone_spans(text: str) -> Iterator[Span]:
    for match in PHONE.finditer(text):
        if has_phone_digit_count(match.group()):
            yield validator_span(match.start(), match.end(), EntityType.PHONE)


def entropy_bits(value: str) -> float:
    counts = Counter(value)
    return -sum(
        count / len(value) * math.log2(count / len(value)) for count in counts.values()
    )


def has_mixed_case_and_digits(value: str) -> bool:
    return (
        any(character.islower() for character in value)
        and any(character.isupper() for character in value)
        and any(character.isdigit() for character in value)
    )


def looks_like_path(value: str) -> bool:
    return value.startswith("/") or "//" in value


def looks_like_secret(value: str) -> bool:
    return (
        len(value) >= MIN_SECRET_LENGTH
        and has_mixed_case_and_digits(value)
        and not HEX_ONLY.fullmatch(value)
        and not looks_like_path(value)
        and entropy_bits(value) >= MIN_SECRET_ENTROPY_BITS
    )


def secret_spans(text: str) -> Iterator[Span]:
    for match in SECRET_CANDIDATE.finditer(text):
        value = match.group().rstrip(SECRET_TRAILING_PUNCTUATION)
        if looks_like_secret(value):
            end = match.start() + len(value)
            yield validator_span(match.start(), end, EntityType.CREDENTIAL)


VALIDATOR_PRIORITY = (
    email_spans,
    card_number_spans,
    phone_spans,
    secret_spans,
)


def without_overlaps(spans_by_priority: Sequence[Span]) -> list[Span]:
    kept: list[Span] = []
    for span in spans_by_priority:
        if not any(span.overlaps(other) for other in kept):
            kept.append(span)
    return sorted(kept, key=lambda span: span.start)


def find_validator_spans(text: str) -> list[Span]:
    found = [span for finder in VALIDATOR_PRIORITY for span in finder(text)]
    return without_overlaps(found)
