import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import (
    DistilBertConfig,
    DistilBertForTokenClassification,
    PreTrainedTokenizerFast,
)

from edgeveil.dataset.augment import openpii_augmented_dir
from edgeveil.detector.labels import ID_TO_LABEL, LABEL_TO_ID
from edgeveil.evaluation.scoring import content_offsets, score_emitted_spans
from edgeveil.evaluation.thresholds import (
    THRESHOLD_BETA,
    THRESHOLD_GRID,
    ThresholdPoint,
    choose_thresholds,
    f_beta,
    tune_thresholds,
    write_threshold_results,
)
from edgeveil.spans import EntityType, GoldSpan, Span, SpanSource

TEXT = "Ada Quill mail ada@example.com now"
GOLD = [GoldSpan(0, 9, EntityType.PERSON), GoldSpan(15, 30, EntityType.EMAIL)]


def as_emitted(spans: list[GoldSpan]) -> list[Span]:
    return [
        Span(span.start, span.end, span.label, 0.9, SpanSource.MODEL) for span in spans
    ]


def test_beta_favors_recall() -> None:
    assert THRESHOLD_BETA == 2.0
    assert f_beta(0.5, 1.0) > f_beta(1.0, 0.5)
    assert f_beta(0.0, 0.0) == 0.0


def test_grid_spans_the_open_unit_interval() -> None:
    assert 0.0 < min(THRESHOLD_GRID) < max(THRESHOLD_GRID) < 1.0


def test_emitted_spans_are_scored_with_strict_iob2(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    offsets = content_offsets(tiny_tokenizer, [TEXT])
    perfect = score_emitted_spans([GOLD], [as_emitted(GOLD)], offsets)
    assert perfect["PERSON"].recall == pytest.approx(1.0)
    assert perfect["EMAIL"].precision == pytest.approx(1.0)
    partial = score_emitted_spans(
        [GOLD], [as_emitted([GoldSpan(0, 3, EntityType.PERSON)])], offsets
    )
    assert partial["PERSON"].recall == pytest.approx(0.0)


def point(threshold: float, precision: float, recall: float) -> ThresholdPoint:
    return ThresholdPoint(
        threshold, precision, recall, f_beta(precision, recall), support=10
    )


def test_choice_maximizes_f_beta_and_breaks_ties_toward_lower_thresholds() -> None:
    curves = {
        EntityType.EMAIL: [
            point(0.3, 0.6, 0.9),
            point(0.5, 0.9, 0.85),
            point(0.7, 0.95, 0.5),
        ],
        EntityType.PHONE: [point(0.2, 0.8, 0.8), point(0.4, 0.8, 0.8)],
    }
    chosen = choose_thresholds(curves)
    assert chosen[EntityType.EMAIL].threshold == 0.5
    assert chosen[EntityType.PHONE].threshold == 0.2


@pytest.fixture
def tiny_run(
    tiny_tokenizer: PreTrainedTokenizerFast, tmp_path: Path
) -> tuple[Path, Path]:
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=len(tiny_tokenizer),
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model_dir = tmp_path / "model"
    DistilBertForTokenClassification(config).save_pretrained(model_dir)
    tiny_tokenizer.save_pretrained(model_dir)
    data_dir = tmp_path / "data"
    augmented = openpii_augmented_dir(data_dir)
    augmented.mkdir(parents=True)
    spans = [
        {"start": span.start, "end": span.end, "label": span.label.value}
        for span in GOLD
    ]
    Dataset.from_list(
        [
            {"uid": index, "origin": "openpii", "source_text": TEXT, "spans": spans}
            for index in range(6)
        ]
    ).to_parquet(str(augmented / "validation.parquet"))
    return model_dir, data_dir


def test_tuning_sweeps_every_type_and_writes_a_results_file(
    tiny_run: tuple[Path, Path], tmp_path: Path
) -> None:
    model_dir, data_dir = tiny_run
    report = tune_thresholds(model_dir, data_dir, device=torch.device("cpu"))
    assert set(report.curves) == set(EntityType)
    assert all(len(points) == len(THRESHOLD_GRID) for points in report.curves.values())
    assert set(report.chosen) == set(EntityType)
    path = tmp_path / "results" / "thresholds.json"
    write_threshold_results(report, path)
    saved = json.loads(path.read_text())
    assert saved["beta"] == THRESHOLD_BETA
    assert set(saved["chosen"]["EMAIL"]) == {
        "threshold",
        "precision",
        "recall",
        "f_beta",
        "support",
    }
    assert "chip" in saved["environment"]
