import json
from pathlib import Path

import pytest

from edgeveil.evaluation.corrections import (
    Correction,
    CorrectionError,
    append_corrections,
    apply_corrections,
    load_corrections,
    propose_corrections,
)
from edgeveil.evaluation.label_lint import LintRule, lint_snippet
from edgeveil.evaluation.label_studio import import_label_studio_export
from edgeveil.evaluation.test_set import LabeledSnippet, load_test_set
from edgeveil.spans import EntityType, GoldSpan

PERSON = EntityType.PERSON


def labeled(text: str, *spans: tuple[str, EntityType]) -> LabeledSnippet:
    gold: list[GoldSpan] = []
    cursor = 0
    for value, label in spans:
        start = text.index(value, cursor)
        gold.append(GoldSpan(start, start + len(value), label))
        cursor = start + len(value)
    return LabeledSnippet(1, text, tuple(gold))


def proposed_values(snippet: LabeledSnippet) -> list[tuple[str, EntityType]]:
    proposals, _ = propose_corrections(snippet)
    return [
        (snippet.text[span.start : span.end], span.label)
        for proposal in proposals
        for span in proposal.new
    ]


def rules(snippet: LabeledSnippet) -> set[LintRule]:
    return {finding.rule for finding in lint_snippet(snippet)}


def test_clean_labels_have_no_findings() -> None:
    snippet = labeled(
        "Ada Quill, ada@example.com, (555) 010-0123, 12 Imaginary Lane",
        ("Ada Quill", PERSON),
        ("ada@example.com", EntityType.EMAIL),
        ("(555) 010-0123", EntityType.PHONE),
        ("12 Imaginary Lane", EntityType.ADDRESS),
    )
    assert lint_snippet(snippet) == []


def test_an_email_without_at_needs_a_decision() -> None:
    snippet = labeled(
        "ship to 12 Imaginary Lane", ("12 Imaginary Lane", EntityType.EMAIL)
    )
    proposals, undecided = propose_corrections(snippet)
    assert proposals == []
    assert [finding.rule for finding in undecided] == [LintRule.EMAIL_HAS_AT]


@pytest.mark.parametrize(
    ("text", "value", "label", "expected"),
    [
        ("Booking for Dr. Ada Quill.", "Dr. Ada Quill", PERSON, "Ada Quill"),
        ("It was Ada's.", "Ada's", PERSON, "Ada"),
        (
            "Auth: Basic YWRhOnF1aWxs",
            "Basic YWRhOnF1aWxs",
            EntityType.CREDENTIAL,
            "YWRhOnF1aWxs",
        ),
        ("mine is DL 0042-118-77", "DL 0042-118-77", EntityType.GOV_ID, "0042-118-77"),
        (
            "Passport No. L0000000",
            "Passport No. L0000000",
            EntityType.GOV_ID,
            "L0000000",
        ),
        ("Phone (555.0125) now", "(555.0125)", EntityType.PHONE, "555.0125"),
        ('pw "Hazel-Moss-0"', '"Hazel-Moss-0"', EntityType.CREDENTIAL, "Hazel-Moss-0"),
        (
            "at 12 Imaginary Lane, Nowhere City, ZZ",
            "12 Imaginary Lane, Nowhere City, ZZ",
            EntityType.ADDRESS,
            "12 Imaginary Lane, Nowhere City",
        ),
        ("code ZIP 00000", "ZIP 00000", EntityType.ADDRESS, "00000"),
    ],
)
def test_words_and_punctuation_at_the_edges_are_trimmed(
    text: str, value: str, label: EntityType, expected: str
) -> None:
    assert proposed_values(labeled(text, (value, label))) == [(expected, label)]


def test_area_code_parentheses_stay_inside_a_phone() -> None:
    snippet = labeled("call (555) 010-0123", ("(555) 010-0123", EntityType.PHONE))
    assert lint_snippet(snippet) == []


def test_a_comma_splits_a_name() -> None:
    snippet = labeled("name: Quill, Ada |", ("Quill, Ada", PERSON))
    assert proposed_values(snippet) == [("Quill", PERSON), ("Ada", PERSON)]


def test_name_parts_joined_by_a_connector_are_one_span() -> None:
    snippet = labeled(
        "holder Ada d/o Ben Quill.", ("Ada", PERSON), ("Ben Quill", PERSON)
    )
    assert proposed_values(snippet) == [("Ada d/o Ben Quill", PERSON)]


def test_names_on_separate_lines_need_a_decision() -> None:
    snippet = labeled(
        "Ada Quill\nBen Stone", ("Ada Quill", PERSON), ("Ben Stone", PERSON)
    )
    _, undecided = propose_corrections(snippet)
    assert [finding.rule for finding in undecided] == [LintRule.PERSON_JOIN]


def test_address_parts_with_an_allowed_gap_are_one_span() -> None:
    snippet = labeled(
        "at 36 Example Crescent, Suite Exampleville",
        ("36 Example Crescent", EntityType.ADDRESS),
        ("Exampleville", EntityType.ADDRESS),
    )
    assert proposed_values(snippet) == [
        ("36 Example Crescent, Suite Exampleville", EntityType.ADDRESS)
    ]


def test_an_address_without_digits_needs_a_decision() -> None:
    snippet = labeled("I moved to Nowhere City.", ("Nowhere City", EntityType.ADDRESS))
    _, undecided = propose_corrections(snippet)
    assert [finding.rule for finding in undecided] == [LintRule.ADDRESS_ANCHOR]


def test_partial_card_numbers_are_unlabeled() -> None:
    snippet = labeled("card ending 4242", ("4242", EntityType.CREDIT_CARD))
    proposals, _ = propose_corrections(snippet)
    assert [(p.old, p.new) for p in proposals] == [(snippet.spans, ())]
    assert proposals[0].rule == LintRule.PARTIAL_CARD.value


def test_masked_ssns_get_their_digits_labeled() -> None:
    snippet = labeled("portal shows SSN ***-**-1234 today")
    assert proposed_values(snippet) == [("1234", EntityType.SSN)]
    done = labeled("portal shows SSN ***-**-1234 today", ("1234", EntityType.SSN))
    assert LintRule.PARTIAL_SSN not in rules(done)


def test_several_rules_on_one_span_give_one_correction() -> None:
    snippet = labeled('owner "Dr. Ada Quill"', ('"Dr. Ada Quill"', PERSON))
    proposals, _ = propose_corrections(snippet)
    assert len(proposals) == 1
    assert proposals[0].rule == "; ".join(
        sorted([LintRule.PERSON_TITLE.value, LintRule.WRAPPING.value])
    )
    assert proposed_values(snippet) == [("Ada Quill", PERSON)]


def test_a_kept_span_counts_as_reviewed() -> None:
    snippet = labeled(
        "ship to 12 Imaginary Lane", ("12 Imaginary Lane", EntityType.EMAIL)
    )
    keep = Correction(1, snippet.spans, snippet.spans, "reviewer kept")
    [corrected] = apply_corrections([snippet], [keep])
    assert lint_snippet(corrected, set(keep.new)) == []


def test_a_correction_that_no_longer_matches_fails() -> None:
    snippet = labeled("Dr. Ada Quill", ("Dr. Ada Quill", PERSON))
    stale = Correction(1, (GoldSpan(0, 5, PERSON),), (), "stale")
    with pytest.raises(CorrectionError, match="no longer matches"):
        apply_corrections([snippet], [stale])


def test_corrections_that_overlap_fail() -> None:
    snippet = labeled("Ada Quill", ("Ada", PERSON), ("Quill", PERSON))
    overlap = Correction(1, (), (GoldSpan(2, 6, PERSON),), "bad")
    with pytest.raises(CorrectionError, match="overlap"):
        apply_corrections([snippet], [overlap])


def test_corrections_round_trip_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / "corrections.jsonl"
    correction = Correction(
        7, (GoldSpan(0, 13, PERSON),), (GoldSpan(4, 13, PERSON),), "titles"
    )
    append_corrections(path, [correction])
    append_corrections(path, [correction])
    assert load_corrections(path) == [correction, correction]
    assert "Ada" not in path.read_text()


def export_task(text: str, start: int, end: int, label: str) -> dict[str, object]:
    region = {
        "type": "labels",
        "value": {
            "start": start,
            "end": end,
            "text": text[start:end],
            "labels": [label],
        },
    }
    return {
        "id": 1,
        "data": {"text": text},
        "annotations": [{"was_cancelled": False, "result": [region]}],
    }


def test_import_fails_on_lint_and_succeeds_after_a_correction(tmp_path: Path) -> None:
    text = "Booking for Dr. Ada Quill."
    export = tmp_path / "export.json"
    export.write_text(json.dumps([export_task(text, 12, 25, "PERSON")]))
    output = tmp_path / "test_set.jsonl"
    corrections = tmp_path / "corrections.jsonl"
    issues = import_label_studio_export([export], output, corrections)
    assert [issue.problem for issue in issues] == [
        f"label lint: {LintRule.PERSON_TITLE.value}"
    ]
    assert not output.exists()
    fix = Correction(
        1, (GoldSpan(12, 25, PERSON),), (GoldSpan(16, 25, PERSON),), "titles"
    )
    append_corrections(corrections, [fix])
    assert import_label_studio_export([export], output, corrections) == []
    [snippet] = load_test_set(output)
    assert snippet.spans == (GoldSpan(16, 25, PERSON),)
