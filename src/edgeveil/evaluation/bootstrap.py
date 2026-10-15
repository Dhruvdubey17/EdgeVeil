import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from edgeveil.detector.metrics import (
    OVERALL_REPORT_KEY,
    EntityScores,
    strict_entity_scores,
)

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 1006
CONFIDENCE_LEVEL = 0.95
OVERALL_NAME = "overall"
MEASURES = ("precision", "recall", "f1")


@dataclass(frozen=True, slots=True)
class Estimate:
    value: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class ScoredType:
    precision: Estimate
    recall: Estimate
    f1: Estimate
    support: int


Tags = Sequence[Sequence[str]]


def public_name(report_key: str) -> str:
    return OVERALL_NAME if report_key == OVERALL_REPORT_KEY else report_key


def measure(scores: EntityScores, name: str) -> float:
    return {"precision": scores.precision, "recall": scores.recall, "f1": scores.f1}[
        name
    ]


def resampled_scores(
    gold: Tags, predicted: Tags, resamples: int, seed: int
) -> list[dict[str, EntityScores]]:
    rng = np.random.default_rng(seed)
    draws: list[dict[str, EntityScores]] = []
    for _ in range(resamples):
        picks = [int(index) for index in rng.integers(0, len(gold), size=len(gold))]
        draws.append(
            strict_entity_scores(
                [gold[index] for index in picks], [predicted[index] for index in picks]
            )
        )
    return draws


def interval(values: Sequence[float]) -> tuple[float, float]:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return math.nan, math.nan
    tail = (1 - CONFIDENCE_LEVEL) / 2 * 100
    low, high = np.percentile(finite, [tail, 100 - tail])
    return float(low), float(high)


def estimate(point: float, draws: Sequence[float]) -> Estimate:
    low, high = interval(draws)
    return Estimate(value=point, low=low, high=high)


def draws_for(
    draws: Sequence[dict[str, EntityScores]], report_key: str, name: str
) -> list[float]:
    return [
        measure(draw[report_key], name) if report_key in draw else math.nan
        for draw in draws
    ]


def score_with_intervals(
    gold: Tags,
    predicted: Tags,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, ScoredType]:
    point = strict_entity_scores(gold, predicted)
    draws = resampled_scores(gold, predicted, resamples, seed)
    scored: dict[str, ScoredType] = {}
    for report_key, scores in point.items():
        estimates = {
            name: estimate(measure(scores, name), draws_for(draws, report_key, name))
            for name in MEASURES
        }
        scored[public_name(report_key)] = ScoredType(
            precision=estimates["precision"],
            recall=estimates["recall"],
            f1=estimates["f1"],
            support=scores.support,
        )
    return scored
