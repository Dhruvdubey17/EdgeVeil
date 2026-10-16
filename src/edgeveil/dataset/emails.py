import random
from collections.abc import Sequence
from dataclasses import dataclass

from edgeveil.spans import EntityType, GoldSpan, LabeledText

EMAIL_DOMAIN_REWRITE_SHARE = 0.25
RESERVED_EXAMPLE_DOMAINS = ("example.com", "example.org", "example.net")
RESERVED_EXAMPLE_SHARE = 0.5
TEST_DOMAIN_WORDS = (
    "lantern",
    "harbor",
    "copper",
    "meadow",
    "falcon",
    "pebble",
    "orbit",
    "violet",
    "summit",
    "willow",
)
TEST_DOMAIN_SHAPES = ("{a}.test", "{a}-{b}.test", "mail.{a}.test", "{a}{b}.test")


@dataclass(frozen=True, slots=True)
class DomainRewrite:
    rows: list[LabeledText]
    emails_seen: int
    domains_rewritten: int


def fictional_domain(rng: random.Random) -> str:
    if rng.random() < RESERVED_EXAMPLE_SHARE:
        return rng.choice(RESERVED_EXAMPLE_DOMAINS)
    first, second = rng.sample(TEST_DOMAIN_WORDS, 2)
    return rng.choice(TEST_DOMAIN_SHAPES).format(a=first, b=second)


def rewritten_email(email: str, rng: random.Random) -> str | None:
    local, separator, _ = email.rpartition("@")
    if not separator or not local:
        return None
    return f"{local}@{fictional_domain(rng)}"


def rewrite_row_email_domains(
    row: LabeledText, rng: random.Random, share: float = EMAIL_DOMAIN_REWRITE_SHARE
) -> LabeledText:
    pieces: list[str] = []
    spans: list[GoldSpan] = []
    cursor = 0
    shift = 0
    for span in sorted(row.spans, key=lambda item: item.start):
        original = row.text[span.start : span.end]
        replacement = None
        if span.label is EntityType.EMAIL and rng.random() < share:
            replacement = rewritten_email(original, rng)
        value = original if replacement is None else replacement
        pieces.append(row.text[cursor : span.start])
        pieces.append(value)
        start = span.start + shift
        spans.append(GoldSpan(start, start + len(value), span.label))
        shift += len(value) - len(original)
        cursor = span.end
    pieces.append(row.text[cursor:])
    return LabeledText(text="".join(pieces), spans=tuple(spans))


def count_emails(row: LabeledText) -> int:
    return sum(span.label is EntityType.EMAIL for span in row.spans)


def changed_email_count(before: LabeledText, after: LabeledText) -> int:
    ordered = sorted(before.spans, key=lambda item: item.start)
    pairs = zip(ordered, after.spans, strict=True)
    return sum(
        old.label is EntityType.EMAIL
        and before.text[old.start : old.end] != after.text[new.start : new.end]
        for old, new in pairs
    )


def rewrite_email_domains(
    rows: Sequence[LabeledText], rng: random.Random
) -> DomainRewrite:
    rewritten = [rewrite_row_email_domains(row, rng) for row in rows]
    return DomainRewrite(
        rows=rewritten,
        emails_seen=sum(count_emails(row) for row in rows),
        domains_rewritten=sum(
            changed_email_count(before, after)
            for before, after in zip(rows, rewritten, strict=True)
        ),
    )
