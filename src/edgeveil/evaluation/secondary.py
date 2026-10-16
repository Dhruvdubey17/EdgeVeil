import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from edgeveil.evaluation.bootstrap import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    OVERALL_NAME,
    Estimate,
    estimate,
)
from edgeveil.evaluation.errors import overlaps
from edgeveil.spans import EntityType, GoldSpan, Span

ENTITY_ORDER = tuple(EntityType)
Counts = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SecondaryScores:
    overlap_precision: Estimate
    overlap_recall: Estimate
    char_recall: Estimate


@dataclass(frozen=True, slots=True)
class MatchCounts:
    predicted: Counts
    predicted_overlapping: Counts
    gold: Counts
    gold_overlapping: Counts
    gold_characters: Counts
    masked_characters: Counts


MATCH_COUNT_FIELDS = len(fields(MatchCounts))


def overlap_hits(
    spans: Sequence[GoldSpan], others: Sequence[GoldSpan]
) -> tuple[list[int], list[int]]:
    totals = [0] * len(ENTITY_ORDER)
    hits = [0] * len(ENTITY_ORDER)
    for span in spans:
        column = ENTITY_ORDER.index(span.label)
        totals[column] += 1
        if any(other.label is span.label and overlaps(span, other) for other in others):
            hits[column] += 1
    return totals, hits


def masked_character_counts(
    gold: Sequence[GoldSpan], emitted: Sequence[Span]
) -> tuple[list[int], list[int]]:
    masked = {index for span in emitted for index in range(span.start, span.end)}
    totals = [0] * len(ENTITY_ORDER)
    covered = [0] * len(ENTITY_ORDER)
    for span in gold:
        column = ENTITY_ORDER.index(span.label)
        totals[column] += span.end - span.start
        covered[column] += sum(index in masked for index in range(span.start, span.end))
    return totals, covered


def match_counts(
    gold_aligned: Sequence[Sequence[GoldSpan]],
    predicted_aligned: Sequence[Sequence[GoldSpan]],
    gold_raw: Sequence[Sequence[GoldSpan]],
    emitted_raw: Sequence[Sequence[Span]],
) -> MatchCounts:
    rows: list[list[list[int]]] = []
    for gold, predicted, raw_gold, raw_emitted in zip(
        gold_aligned, predicted_aligned, gold_raw, emitted_raw, strict=True
    ):
        predicted_totals, predicted_hits = overlap_hits(predicted, gold)
        gold_totals, gold_hits = overlap_hits(gold, predicted)
        characters, masked = masked_character_counts(raw_gold, raw_emitted)
        rows.append(
            [
                predicted_totals,
                predicted_hits,
                gold_totals,
                gold_hits,
                characters,
                masked,
            ]
        )
    table = np.array(rows, dtype=np.int64).reshape(
        len(rows), MATCH_COUNT_FIELDS, len(ENTITY_ORDER)
    )
    return MatchCounts(*(table[:, field, :] for field in range(MATCH_COUNT_FIELDS)))


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def ratio_estimate(
    numerators: Counts, denominators: Counts, picks: NDArray[np.int64]
) -> Estimate:
    point = ratio(float(numerators.sum()), float(denominators.sum()))
    resampled_numerators = numerators[picks].sum(axis=1)
    resampled_denominators = denominators[picks].sum(axis=1)
    draws = [
        ratio(float(top), float(bottom))
        for top, bottom in zip(
            resampled_numerators, resampled_denominators, strict=True
        )
    ]
    return estimate(point, draws)


def per_snippet(values: Counts, column: int | None) -> Counts:
    if column is None:
        return values.sum(axis=1)
    return values[:, column]


def scores_for(
    counts: MatchCounts, column: int | None, picks: NDArray[np.int64]
) -> SecondaryScores:
    return SecondaryScores(
        overlap_precision=ratio_estimate(
            per_snippet(counts.predicted_overlapping, column),
            per_snippet(counts.predicted, column),
            picks,
        ),
        overlap_recall=ratio_estimate(
            per_snippet(counts.gold_overlapping, column),
            per_snippet(counts.gold, column),
            picks,
        ),
        char_recall=ratio_estimate(
            per_snippet(counts.masked_characters, column),
            per_snippet(counts.gold_characters, column),
            picks,
        ),
    )


def secondary_scores(
    counts: MatchCounts,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, SecondaryScores]:
    snippets = counts.gold.shape[0]
    picks = np.random.default_rng(seed).integers(
        0, snippets, size=(resamples, snippets)
    )
    present = (counts.gold.sum(axis=0) + counts.predicted.sum(axis=0)) > 0
    scored = {
        entity.value: scores_for(counts, column, picks)
        for column, entity in enumerate(ENTITY_ORDER)
        if present[column]
    }
    scored[OVERALL_NAME] = scores_for(counts, None, picks)
    return scored
