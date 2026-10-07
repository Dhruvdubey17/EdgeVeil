import json
import platform
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import transformers
from datasets import Dataset
from seqeval.metrics import classification_report
from seqeval.metrics.v1 import UndefinedMetricWarning
from torch.nn import functional
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from edgeveil.dataset.augment import openpii_augmented_dir
from edgeveil.dataset.openpii import OPENPII_REVISION
from edgeveil.detector.alignment import token_label_ids
from edgeveil.detector.labels import (
    BIO_LABELS,
    ID_TO_LABEL,
    IGNORED_LABEL_ID,
    LABEL_TO_ID,
)
from edgeveil.detector.presets import PRESETS, ModelPreset, TrainingConfig
from edgeveil.detector.windows import encode_windows
from edgeveil.spans import EntityType, GoldSpan

TRAINING_SEED = 1006
GRADIENT_ACCUMULATION_STEPS = 1
CLASS_WEIGHT_POWER = 0.5
MAX_CLASS_WEIGHT = 10.0
EVAL_BATCH_SIZE = 64
LOGGING_STEPS = 200
CHECKPOINTS_KEPT = 1
RUN_RECORD_NAME = "training_run.json"
OVERALL_METRICS_KEY = "micro avg"


@dataclass(frozen=True, slots=True)
class SplitDatasets:
    train: Dataset
    validation: Dataset


@dataclass(frozen=True, slots=True)
class RunOptions:
    output_dir: Path
    max_steps: int | None = None
    use_cpu: bool = False


def spans_from_records(records: Iterable[Mapping[str, Any]]) -> list[GoldSpan]:
    return [
        GoldSpan(record["start"], record["end"], EntityType(record["label"]))
        for record in records
    ]


def encode_training_batch(
    batch: Mapping[str, list[Any]], tokenizer: PreTrainedTokenizerBase
) -> dict[str, list[list[int]]]:
    texts: list[str] = batch["source_text"]
    encoded: dict[str, list[list[int]]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for windows, span_records in zip(
        encode_windows(tokenizer, texts), batch["spans"], strict=True
    ):
        spans = spans_from_records(span_records)
        for window in windows:
            encoded["input_ids"].append(window.input_ids)
            encoded["attention_mask"].append([1] * len(window.input_ids))
            encoded["labels"].append(token_label_ids(spans, window.offsets))
    return encoded


def windowed_split(
    data_dir: Path, split: str, tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    augmented_dir = openpii_augmented_dir(data_dir)
    rows = Dataset.from_parquet(
        str(augmented_dir / f"{split}.parquet"), cache_dir=str(augmented_dir / "cache")
    )
    return rows.map(
        encode_training_batch,
        batched=True,
        remove_columns=rows.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )


def class_weights(label_rows: Iterable[Sequence[int]]) -> list[float]:
    labels = np.concatenate([np.asarray(row, dtype=np.int64) for row in label_rows])
    counted = labels[labels != IGNORED_LABEL_ID]
    counts = np.bincount(counted, minlength=len(BIO_LABELS)).astype(np.float64)
    balanced = counted.size / (len(BIO_LABELS) * np.maximum(counts, 1.0))
    return np.minimum(balanced**CLASS_WEIGHT_POWER, MAX_CLASS_WEIGHT).tolist()


def weighted_token_loss(weights: Sequence[float]) -> Callable[..., torch.Tensor]:
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    def loss(
        outputs: Any, labels: torch.Tensor, num_items_in_batch: Any = None
    ) -> torch.Tensor:
        logits = outputs.logits
        return functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            weight=weight_tensor.to(logits.device),
            ignore_index=IGNORED_LABEL_ID,
        )

    return loss


def most_likely_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def label_sequences(
    predictions: Any, label_ids: Any
) -> tuple[list[list[str]], list[list[str]]]:
    gold: list[list[str]] = []
    predicted: list[list[str]] = []
    for predicted_row, gold_row in zip(
        np.asarray(predictions), np.asarray(label_ids), strict=True
    ):
        kept = gold_row != IGNORED_LABEL_ID
        gold.append([ID_TO_LABEL[int(label)] for label in gold_row[kept]])
        predicted.append([ID_TO_LABEL[int(label)] for label in predicted_row[kept]])
    return gold, predicted


def entity_metrics(evaluation: EvalPrediction) -> dict[str, float]:
    gold, predicted = label_sequences(evaluation.predictions, evaluation.label_ids)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        report = cast(
            dict[str, dict[str, float]],
            classification_report(gold, predicted, output_dict=True),
        )
    metrics: dict[str, float] = {}
    for entity in [*(entity.value for entity in EntityType), OVERALL_METRICS_KEY]:
        scores = report.get(entity)
        if scores is None:
            continue
        name = "overall" if entity == OVERALL_METRICS_KEY else entity
        for measure in ("precision", "recall", "f1-score"):
            metrics[f"{name}_{measure.removesuffix('-score')}"] = float(scores[measure])
    return metrics


def training_arguments(
    config: TrainingConfig, options: RunOptions
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(options.output_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        max_steps=options.max_steps or -1,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=config.warmup_share,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=CHECKPOINTS_KEPT,
        logging_steps=LOGGING_STEPS,
        seed=TRAINING_SEED,
        data_seed=TRAINING_SEED,
        dataloader_pin_memory=False,
        use_cpu=options.use_cpu,
        report_to="none",
    )


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    datasets: SplitDatasets,
    config: TrainingConfig,
    options: RunOptions,
) -> Trainer:
    return Trainer(
        model=model,
        args=training_arguments(config, options),
        train_dataset=datasets.train,
        eval_dataset=datasets.validation,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_loss_func=weighted_token_loss(class_weights(datasets.train["labels"])),
        compute_metrics=entity_metrics,
        preprocess_logits_for_metrics=most_likely_labels,
    )


def load_base_model(
    config: TrainingConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    set_seed(TRAINING_SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, revision=config.revision
    )
    model = AutoModelForTokenClassification.from_pretrained(
        config.base_model,
        revision=config.revision,
        num_labels=len(BIO_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    return model, tokenizer


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    preset: str
    base_model: str
    base_model_revision: str
    dataset_revision: str
    training_seed: int
    config: dict[str, Any]
    device: str
    platform: str
    torch_version: str
    transformers_version: str
    train_windows: int
    validation_windows: int
    class_weights: list[float]
    training_seconds: float
    validation_metrics: dict[str, float]


def train_detector(
    data_dir: Path, models_dir: Path, preset: ModelPreset, max_steps: int | None = None
) -> TrainingRecord:
    config = PRESETS[preset]
    model, tokenizer = load_base_model(config)
    datasets = SplitDatasets(
        train=windowed_split(data_dir, "train", tokenizer),
        validation=windowed_split(data_dir, "validation", tokenizer),
    )
    output_dir = models_dir / preset.value
    options = RunOptions(output_dir=output_dir / "checkpoints", max_steps=max_steps)
    trainer = build_trainer(model, tokenizer, datasets, config, options)
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    trainer.save_model(str(output_dir))
    record = TrainingRecord(
        preset=preset.value,
        base_model=config.base_model,
        base_model_revision=config.revision,
        dataset_revision=OPENPII_REVISION,
        training_seed=TRAINING_SEED,
        config=asdict(config),
        device=str(trainer.args.device),
        platform=platform.platform(),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        train_windows=len(datasets.train),
        validation_windows=len(datasets.validation),
        class_weights=class_weights(datasets.train["labels"]),
        training_seconds=round(elapsed, 1),
        validation_metrics=trainer.evaluate(),
    )
    (output_dir / RUN_RECORD_NAME).write_text(json.dumps(asdict(record), indent=2))
    return record
