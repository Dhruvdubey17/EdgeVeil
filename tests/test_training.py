import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import (
    DistilBertConfig,
    DistilBertForTokenClassification,
    EvalPrediction,
    PreTrainedTokenizerFast,
)

from edgeveil.detector.alignment import decode_label_ids
from edgeveil.detector.labels import (
    BIO_LABELS,
    ID_TO_LABEL,
    IGNORED_LABEL_ID,
    LABEL_TO_ID,
)
from edgeveil.detector.presets import PRESETS, ModelPreset
from edgeveil.detector.training import (
    MAX_CLASS_WEIGHT,
    RunOptions,
    SplitDatasets,
    build_trainer,
    class_weights,
    encode_training_batch,
    entity_metrics,
    planned_steps,
    weighted_token_loss,
)
from edgeveil.detector.windows import encode_windows
from edgeveil.spans import EntityType, GoldSpan

TEXT = "Ada Quill mail ada@example.com now"
SPANS = [
    {"start": 0, "end": 9, "label": "PERSON"},
    {"start": 15, "end": 30, "label": "EMAIL"},
]


def test_presets_pin_a_revision_for_each_base_model() -> None:
    assert set(PRESETS) == set(ModelPreset)
    assert all(len(config.revision) == 40 for config in PRESETS.values())


def test_class_weights_favor_rare_labels_and_ignore_padding() -> None:
    outside, begin = LABEL_TO_ID["O"], LABEL_TO_ID["B-EMAIL"]
    weights = class_weights([[outside] * 99 + [begin], [IGNORED_LABEL_ID] * 50])
    assert len(weights) == len(BIO_LABELS)
    assert weights[begin] > weights[outside]
    assert max(weights) <= MAX_CLASS_WEIGHT


def test_training_batch_labels_decode_back_to_the_spans(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    encoded = encode_training_batch(
        {"source_text": [TEXT], "spans": [SPANS]}, tokenizer=tiny_tokenizer
    )
    [labels] = encoded["labels"]
    [window] = encode_windows(tiny_tokenizer, [TEXT])[0]
    assert encoded["input_ids"] == [window.input_ids]
    assert decode_label_ids(labels, window.offsets) == [
        GoldSpan(0, 9, EntityType.PERSON),
        GoldSpan(15, 30, EntityType.EMAIL),
    ]


def test_weighted_loss_ignores_padding_and_applies_weights() -> None:
    logits = torch.zeros(1, 3, len(BIO_LABELS))
    labels = torch.tensor([[0, 1, IGNORED_LABEL_ID]])
    weights = [1.0] * len(BIO_LABELS)
    weights[1] = 3.0
    loss = weighted_token_loss(weights)(
        SimpleNamespace(logits=logits), labels, num_items_in_batch=None
    )
    assert loss.item() == pytest.approx(math.log(len(BIO_LABELS)))


def test_entity_metrics_score_perfect_predictions_as_one() -> None:
    ids = [
        LABEL_TO_ID[label] for label in ("O", "B-PERSON", "I-PERSON", "O", "B-EMAIL")
    ]
    labels = [[IGNORED_LABEL_ID, *ids, IGNORED_LABEL_ID]]
    predictions = [[0, *ids, 0]]
    metrics = entity_metrics(
        EvalPrediction(predictions=np.array(predictions), label_ids=np.array(labels))
    )
    assert metrics["PERSON_f1"] == pytest.approx(1.0)
    assert metrics["EMAIL_recall"] == pytest.approx(1.0)
    assert metrics["overall_f1"] == pytest.approx(1.0)


def tiny_model(tokenizer: PreTrainedTokenizerFast) -> DistilBertForTokenClassification:
    config = DistilBertConfig(
        vocab_size=len(tokenizer),
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    return DistilBertForTokenClassification(config)


def test_two_training_steps_run_and_save_a_token_classifier(
    tiny_tokenizer: PreTrainedTokenizerFast, tmp_path: Path
) -> None:
    rows = Dataset.from_dict({"source_text": [TEXT] * 8, "spans": [SPANS] * 8})
    encoded = rows.map(
        encode_training_batch,
        batched=True,
        remove_columns=rows.column_names,
        fn_kwargs={"tokenizer": tiny_tokenizer},
    )
    trainer = build_trainer(
        tiny_model(tiny_tokenizer),
        tiny_tokenizer,
        SplitDatasets(train=encoded, validation=encoded),
        PRESETS[ModelPreset.DISTILBERT],
        RunOptions(output_dir=tmp_path / "run", max_steps=2, use_cpu=True),
    )
    assert trainer.loss_is_scaled_for_ga is False
    result = trainer.train()
    assert math.isfinite(result.training_loss)
    trainer.save_model(str(tmp_path / "model"))
    saved = json.loads((tmp_path / "model" / "config.json").read_text())
    assert saved["id2label"]["0"] == "O"


def test_entity_metrics_use_strict_iob2_so_an_orphan_inside_tag_is_wrong() -> None:
    gold = [
        [IGNORED_LABEL_ID, LABEL_TO_ID["O"], LABEL_TO_ID["B-PERSON"], IGNORED_LABEL_ID]
    ]
    predicted = [[0, LABEL_TO_ID["O"], LABEL_TO_ID["I-PERSON"], 0]]
    metrics = entity_metrics(
        EvalPrediction(predictions=np.array(predicted), label_ids=np.array(gold))
    )
    assert metrics["PERSON_recall"] == pytest.approx(0.0)


def test_planned_steps_round_each_epoch_up() -> None:
    config = PRESETS[ModelPreset.DISTILBERT]
    assert planned_steps(config.effective_batch_size + 1, config) == 2 * config.epochs
