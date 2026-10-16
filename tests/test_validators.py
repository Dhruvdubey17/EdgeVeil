from itertools import pairwise

import pytest

from edgeveil.spans import EntityType, Span, SpanSource
from edgeveil.validators.combiner import combine_spans
from edgeveil.validators.finders import (
    find_validator_spans,
    is_valid_card_number,
    passes_luhn,
)

SECRET = "Zq7Lm2Xp9Rt4Wv8Ks3Nd6Hb1"


def found(text: str) -> list[tuple[str, EntityType]]:
    return [
        (text[span.start : span.end], span.label) for span in find_validator_spans(text)
    ]


def model(start: int, end: int, label: EntityType) -> Span:
    return Span(start, end, label, 0.9, SpanSource.MODEL)


@pytest.mark.parametrize(
    "number", ["4111111111111111", "5555555555554444", "378282246310005"]
)
def test_processor_test_numbers_pass_luhn(number: str) -> None:
    assert passes_luhn(number)


def test_one_changed_digit_fails_luhn() -> None:
    assert not passes_luhn("4111111111111112")


@pytest.mark.parametrize(
    "card", ["4111 1111 1111 1111", "5555-5555-5555-4444", "378282246310005"]
)
def test_cards_are_found_with_any_single_separator_style(card: str) -> None:
    assert found(f"Card {card} on file.") == [(card, EntityType.CREDIT_CARD)]


def test_mixed_separators_and_luhn_failures_are_not_cards() -> None:
    assert found("Card 4111 1111-1111 1111.") == []
    assert found("Ref 4111 1111 1111 1112.") == []


def test_card_followed_by_expiry_keeps_only_the_number() -> None:
    assert found("4111 1111 1111 1111 12/29") == [
        ("4111 1111 1111 1111", EntityType.CREDIT_CARD)
    ]


@pytest.mark.parametrize(
    "email", ["ada.quill@example.com", "a+tag@mail.harbor.test", "b_c@example.org"]
)
def test_emails_are_found_without_trailing_punctuation(email: str) -> None:
    assert found(f"Write to <{email}>.") == [(email, EntityType.EMAIL)]


def test_double_dots_and_missing_domains_are_not_emails() -> None:
    assert found("ada..quill@example.com") == []
    assert found("ada@localhost") == []


@pytest.mark.parametrize(
    "phone",
    ["(202) 555-0143", "202-555-0143", "202.555.0143", "+1 202 555 0143", "555-0143"],
)
def test_phones_are_found_in_common_shapes(phone: str) -> None:
    assert found(f"Call {phone} today.") == [(phone, EntityType.PHONE)]


@pytest.mark.parametrize(
    "text", ["SSN 900-12-3456", "on 2026-10-16", "version v2.14.1", "price 1,299.00"]
)
def test_ssns_dates_versions_and_prices_are_not_phones(text: str) -> None:
    assert found(text) == []


def test_high_entropy_tokens_are_credentials() -> None:
    assert found(f"api_key={SECRET}.") == [(SECRET, EntityType.CREDENTIAL)]


@pytest.mark.parametrize(
    "text",
    [
        "commit 3f786850e387550fdab836ed7e6dc881de23001b",
        "id 123e4567-e89b-12d3-a456-426614174000",
        "see configuration_settings_file_name",
        "path /usr/local/lib/Python3.13/site-packages",
    ],
)
def test_hashes_uuids_words_and_paths_are_not_credentials(text: str) -> None:
    assert found(text) == []


def test_non_breaking_spaces_bound_emails_and_phones() -> None:
    text = "mail\u00a0ada@example.com\u00a0or (202)\u00a0555-0143"
    assert found(text) == [
        ("ada@example.com", EntityType.EMAIL),
        ("(202)\u00a0555-0143", EntityType.PHONE),
    ]


def test_card_number_hard_check_counts_digits() -> None:
    assert is_valid_card_number("4111-1111-1111-1111")
    assert not is_valid_card_number("4111")


def test_combiner_vetoes_cards_failing_luhn_and_short_phones() -> None:
    text = "Ref 4111 1111 1111 1112 ext 0143"
    spans = [model(4, 23, EntityType.CREDIT_CARD), model(28, 32, EntityType.PHONE)]
    assert combine_spans(text, spans, []) == []


def test_validator_boundaries_replace_a_same_type_model_span() -> None:
    text = "mail ada@example.com now"
    validator = find_validator_spans(text)
    combined = combine_spans(text, [model(5, 9, EntityType.EMAIL)], validator)
    assert combined == validator


def test_model_span_wins_when_it_strictly_contains_another_type() -> None:
    text = "ship to 12 Imaginary Lane 202-555-0143 Nowhere"
    address = model(8, 46, EntityType.ADDRESS)
    combined = combine_spans(text, [address], find_validator_spans(text))
    assert combined == [address]


def test_validator_wins_over_a_contained_model_span_of_another_type() -> None:
    text = "mail ada.quill@example.com now"
    combined = combine_spans(
        text, [model(5, 13, EntityType.PERSON)], find_validator_spans(text)
    )
    assert [(span.label, span.source) for span in combined] == [
        (EntityType.EMAIL, SpanSource.VALIDATOR)
    ]


def test_a_partly_overlapping_model_span_keeps_the_part_outside() -> None:
    text = "Regards, Ada Quill ada.quill@example.com"
    combined = combine_spans(
        text, [model(9, 22, EntityType.PERSON)], find_validator_spans(text)
    )
    assert [(text[span.start : span.end], span.label) for span in combined] == [
        ("Ada Quill", EntityType.PERSON),
        ("ada.quill@example.com", EntityType.EMAIL),
    ]


def test_combined_spans_never_overlap() -> None:
    text = f"key {SECRET} card 4111 1111 1111 1111 call 202-555-0143"
    spans = [model(0, 20, EntityType.PERSON), model(30, 40, EntityType.GOV_ID)]
    combined = combine_spans(text, spans, find_validator_spans(text))
    assert all(not left.overlaps(right) for left, right in pairwise(combined))
