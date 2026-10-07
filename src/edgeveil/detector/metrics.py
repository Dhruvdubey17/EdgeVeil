import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from seqeval.metrics import classification_report
from seqeval.metrics.v1 import UndefinedMetricWarning
from seqeval.scheme import IOB2

from edgeveil.spans import EntityType

SEQEVAL_MODE = "strict"
OVERALL_REPORT_KEY = "micro avg"


@dataclass(frozen=True, slots=True)
class EntityScores:
    precision: float
    recall: float
    f1: float
    support: int


def strict_entity_scores(
    gold: Sequence[Sequence[str]], predicted: Sequence[Sequence[str]]
) -> dict[str, EntityScores]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        report = cast(
            dict[str, dict[str, float]],
            classification_report(
                [list(row) for row in gold],
                [list(row) for row in predicted],
                output_dict=True,
                mode=SEQEVAL_MODE,
                scheme=IOB2,
            ),
        )
    names = [entity.value for entity in EntityType] + [OVERALL_REPORT_KEY]
    return {
        name: EntityScores(
            precision=float(report[name]["precision"]),
            recall=float(report[name]["recall"]),
            f1=float(report[name]["f1-score"]),
            support=int(report[name]["support"]),
        )
        for name in names
        if name in report
    }
