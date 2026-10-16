import random
import re

import pytest

from edgeveil.dataset.emails import (
    EMAIL_DOMAIN_REWRITE_SHARE,
    fictional_domain,
    rewrite_email_domains,
    rewrite_row_email_domains,
)
from edgeveil.spans import EntityType, GoldSpan, LabeledText

RESERVED_DOMAIN = re.compile(r"(example\.(com|org|net)|[a-z0-9.-]+\.test)")
TEXT = "Mail ada.quill@lanternmail.com or call 555-0142, cc bo@corp-mail.net."
ROW = LabeledText(
    text=TEXT,
    spans=(
        GoldSpan(5, 30, EntityType.EMAIL),
        GoldSpan(39, 47, EntityType.PHONE),
        GoldSpan(52, 68, EntityType.EMAIL),
    ),
)


def span_texts(row: LabeledText) -> list[tuple[str, EntityType]]:
    return [(row.text[span.start : span.end], span.label) for span in row.spans]


@pytest.mark.parametrize("seed", range(100))
def test_fictional_domains_are_reserved_names(seed: int) -> None:
    assert RESERVED_DOMAIN.fullmatch(fictional_domain(random.Random(seed)))


def test_rewriting_keeps_local_parts_and_shifts_later_spans() -> None:
    result = rewrite_row_email_domains(ROW, random.Random(1), share=1.0)
    texts = span_texts(result)
    assert texts[0][0].startswith("ada.quill@")
    assert texts[2][0].startswith("bo@")
    assert texts[1] == ("555-0142", EntityType.PHONE)
    for email, label in (texts[0], texts[2]):
        assert label is EntityType.EMAIL
        assert RESERVED_DOMAIN.fullmatch(email.split("@", 1)[1])
    assert result.text.endswith(".")


def test_share_zero_leaves_the_row_alone() -> None:
    assert rewrite_row_email_domains(ROW, random.Random(1), share=0.0) == ROW


def test_split_rewrite_hits_the_target_share() -> None:
    rows = [ROW] * 500
    result = rewrite_email_domains(rows, random.Random(2))
    assert result.emails_seen == 1000
    assert result.domains_rewritten / result.emails_seen == pytest.approx(
        EMAIL_DOMAIN_REWRITE_SHARE, abs=0.05
    )


def test_email_without_at_sign_is_left_alone() -> None:
    row = LabeledText(
        text="contact adaquill", spans=(GoldSpan(8, 16, EntityType.EMAIL),)
    )
    assert rewrite_row_email_domains(row, random.Random(3), share=1.0) == row
