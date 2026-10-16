import numpy as np
import pytest

from edgeveil.detector.labels import BIO_LABELS, OUTSIDE_ID, begin_id
from edgeveil.detector.windows import Window
from edgeveil.evaluation.parity import compare_logits
from edgeveil.spans import EntityType

WINDOW = Window(
    input_ids=[1, 2, 3, 4, 5, 6],
    offsets=[(0, 0), (0, 1), (2, 3), (4, 5), (6, 7), (0, 0)],
)


def one_hot(label_ids: list[int]) -> np.ndarray:
    logits = np.zeros((len(label_ids), len(BIO_LABELS)), dtype=np.float32)
    logits[np.arange(len(label_ids)), label_ids] = 1.0
    return logits


def test_agreement_ignores_special_tokens_and_counts_entity_tokens() -> None:
    person = begin_id(EntityType.PERSON)
    email = begin_id(EntityType.EMAIL)
    reference = one_hot([person, person, person, OUTSIDE_ID, OUTSIDE_ID, person])
    candidate = one_hot([email, person, email, OUTSIDE_ID, person, email])
    report = compare_logits([reference], [candidate], [WINDOW])
    assert report.tokens == 4
    assert report.token_agreement == pytest.approx(2 / 4)
    assert report.reference_entity_tokens == 2
    assert report.entity_token_agreement == pytest.approx(1 / 2)
    assert report.max_abs_logit_difference == pytest.approx(1.0)
