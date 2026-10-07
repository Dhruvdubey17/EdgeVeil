import random
import re
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from edgeveil.dataset.credentials import (
    API_KEY_SHAPES,
    CREDENTIAL_ROW_FRACTION,
    HARD_NEGATIVE_ROW_FRACTION,
    HARD_NEGATIVE_TEMPLATES,
    STANDALONE_TEMPLATES,
    LabeledText,
    Origin,
    augment_split,
    generate_api_key,
    generate_hard_negative,
    generate_password,
    generate_token,
    insert_snippet,
    insertion_points,
    render_template,
)
from edgeveil.spans import EntityType, GoldSpan

REPOSITORY_ROOT = Path(__file__).parent.parent
PROVIDER_KEY_PATTERNS = {
    "aws": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github": re.compile(r"ghp_[A-Za-z0-9]{36}"),
    "openai": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{40,}"),
    "anthropic": re.compile(r"sk-ant-api03-[A-Za-z0-9_-]{80,}"),
    "stripe": re.compile(r"sk_live_[A-Za-z0-9]{24,}"),
    "google": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "slack": re.compile(r"xoxb-[0-9]{10,}-[0-9]{10,}-[A-Za-z0-9]{20,}"),
}
JWT_PATTERN = re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
WHITESPACE = re.compile(r"\s")


def email_row(text: str, value: str) -> LabeledText:
    start = text.index(value)
    return LabeledText(
        text=text, spans=(GoldSpan(start, start + len(value), EntityType.EMAIL),)
    )


def span_texts(row: LabeledText) -> list[tuple[str, EntityType]]:
    return [(row.text[span.start : span.end], span.label) for span in row.spans]


def test_every_provider_shape_produces_its_real_format() -> None:
    rng = random.Random(1)
    produced = [generate_api_key(rng, shape) for shape in API_KEY_SHAPES]
    for name, pattern in PROVIDER_KEY_PATTERNS.items():
        assert any(pattern.fullmatch(key) for key in produced), name


def test_tokens_include_jwts() -> None:
    rng = random.Random(2)
    assert any(JWT_PATTERN.fullmatch(generate_token(rng)) for _ in range(50))


@pytest.mark.parametrize("seed", range(20))
def test_generated_secrets_contain_no_whitespace(seed: int) -> None:
    rng = random.Random(seed)
    secrets = [generate_password(rng), generate_token(rng)]
    secrets += [generate_api_key(rng, shape) for shape in API_KEY_SHAPES]
    assert not any(WHITESPACE.search(secret) for secret in secrets)


def test_repository_holds_no_literal_provider_keys() -> None:
    paths = [
        path
        for folder in ("src", "tests", "docs")
        for path in (REPOSITORY_ROOT / folder).rglob("*")
    ]
    offenders = [
        f"{path.relative_to(REPOSITORY_ROOT)}: {name}"
        for path in paths
        if path.is_file() and path.suffix in {".py", ".md", ".xml", ".toml"}
        for name, pattern in PROVIDER_KEY_PATTERNS.items()
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_rendered_template_labels_only_the_secret_values() -> None:
    rng = random.Random(3)
    for template in STANDALONE_TEMPLATES:
        rendered = render_template(template, rng)
        assert rendered.spans
        for span in rendered.spans:
            value = rendered.text[span.start : span.end]
            assert span.label is EntityType.CREDENTIAL
            assert value == value.strip()
            assert value.strip("\"'") == value
            assert not value.startswith("Bearer")


@pytest.mark.parametrize("template", HARD_NEGATIVE_TEMPLATES)
def test_hard_negatives_have_no_spans(template: str) -> None:
    rendered = generate_hard_negative(random.Random(4), template)
    assert rendered.spans == ()
    assert "{" not in rendered.text


def test_insertion_points_avoid_the_inside_of_spans() -> None:
    row = email_row("Hi. Mail ada@example.com. Bye", "ada@example.com")
    points = insertion_points(row)
    assert 0 in points
    assert len(row.text) in points
    assert all(
        not span.start < point < span.end for point in points for span in row.spans
    )


def test_inserting_a_snippet_shifts_later_spans() -> None:
    row = email_row("Hi. Mail ada@example.com now", "ada@example.com")
    snippet = LabeledText(
        text="My password is x1.", spans=(GoldSpan(15, 17, EntityType.CREDENTIAL),)
    )
    result = insert_snippet(row, snippet, position=4)
    assert result.text == "Hi. My password is x1. Mail ada@example.com now"
    assert span_texts(result) == [
        ("x1", EntityType.CREDENTIAL),
        ("ada@example.com", EntityType.EMAIL),
    ]


def test_inserting_at_the_end_adds_a_space() -> None:
    row = email_row("Mail ada@example.com", "ada@example.com")
    snippet = LabeledText(
        text="Key: k9.", spans=(GoldSpan(5, 7, EntityType.CREDENTIAL),)
    )
    result = insert_snippet(row, snippet, position=len(row.text))
    assert result.text == "Mail ada@example.com Key: k9."
    assert span_texts(result)[1] == ("k9", EntityType.CREDENTIAL)


segments = st.lists(
    st.tuples(
        st.text(alphabet="ab .\n", max_size=5),
        st.text(alphabet="xyz", min_size=1, max_size=4),
    ),
    max_size=6,
)


@given(segments=segments, seed=st.integers(min_value=0, max_value=10_000))
def test_augmenting_keeps_original_span_texts(
    segments: list[tuple[str, str]], seed: int
) -> None:
    text = ""
    spans: list[GoldSpan] = []
    for gap, value in segments:
        text += gap
        spans.append(GoldSpan(len(text), len(text) + len(value), EntityType.EMAIL))
        text += value
    rows = [LabeledText(text=text or "a", spans=tuple(spans) if text else ())] * 30
    result = augment_split(rows, random.Random(seed))
    original = sorted(span_texts(rows[0]))
    for row, origin in zip(result.rows, result.origins, strict=True):
        if origin is Origin.CREDENTIAL_TEMPLATE:
            continue
        kept = [item for item in span_texts(row) if item[1] is EntityType.EMAIL]
        assert sorted(kept) == original
        assert all(
            left.end <= right.start
            for left, right in zip(row.spans, row.spans[1:], strict=False)
        )


def test_augmentation_hits_the_target_fractions() -> None:
    rows = [
        email_row(f"Row {index}. Mail ada@example.com.", "ada@example.com")
        for index in range(1000)
    ]
    result = augment_split(rows, random.Random(5))
    with_credentials = sum(
        any(span.label is EntityType.CREDENTIAL for span in row.spans)
        for row in result.rows
    )
    assert with_credentials / len(result.rows) == pytest.approx(
        CREDENTIAL_ROW_FRACTION, abs=0.005
    )
    negatives = result.origins.count(Origin.OPENPII_WITH_HARD_NEGATIVE)
    assert negatives / len(rows) == pytest.approx(HARD_NEGATIVE_ROW_FRACTION, abs=0.005)


def test_augmentation_is_deterministic_for_a_seed() -> None:
    rows = [
        email_row(f"Row {index}. Mail ada@example.com.", "ada@example.com")
        for index in range(200)
    ]
    first = augment_split(rows, random.Random(6))
    second = augment_split(rows, random.Random(6))
    assert [row.text for row in first.rows] == [row.text for row in second.rows]


def test_labeled_text_repr_hides_text() -> None:
    assert "example.com" not in repr(
        email_row("Mail ada@example.com", "ada@example.com")
    )
