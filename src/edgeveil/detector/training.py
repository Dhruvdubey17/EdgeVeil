import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch.nn import functional
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from edgeveil.bench.environment import chip_name, environment_record
from edgeveil.dataset.augment import openpii_augmented_dir
from edgeveil.dataset.openpii import OPENPII_REVISION
from edgeveil.detector.alignment import token_label_ids
from edgeveil.detector.labels import (
    BIO_LABELS,
    ID_TO_LABEL,
    IGNORED_LABEL_ID,
    LABEL_TO_ID,
)
from edgeveil.detector.metrics import OVERALL_REPORT_KEY, strict_entity_scores
from edgeveil.detector.presets import (
    PRESETS,
    ModelPreset,
    TrainingConfig,
    with_device_batch,
)
from edgeveil.detector.windows import encode_windows
from edgeveil.spans import EntityType, GoldSpan

TRAINING_SEED = 1006
CLASS_WEIGHT_POWER = 0.5
MAX_CLASS_WEIGHT = 10.0
EVAL_BATCH_SIZE = 16
EMPTY_CACHE_EVERY_STEPS = 50
DEFAULT_LOGGING_STEPS = 100
CHECKPOINTS_KEPT = 2
LENGTH_COLUMN = "length"
SAMPLING_STRATEGY = "group_by_length"
RUN_RECORD_NAME = "training_run.json"


@dataclass(frozen=True, slots=True)
class SplitDatasets:
    train: Dataset
    validation: Dataset


@dataclass(frozen=True, slots=True)
class RunOptions:
    output_dir: Path
    max_steps: int | None = None
    use_cpu: bool = False
    logging_steps: int = DEFAULT_LOGGING_STEPS
    resume: bool = False
    device_batch_size: int | None = None


def spans_from_records(records: Iterable[Mapping[str, Any]]) -> list[GoldSpan]:
    return [
        GoldSpan(record["start"], record["end"], EntityType(record["label"]))
        for record in records
    ]


def encode_training_batch(
    batch: Mapping[str, list[Any]], tokenizer: PreTrainedTokenizerFast
) -> dict[str, list[Any]]:
    texts: list[str] = batch["source_text"]
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    for windows, span_records in zip(
        encode_windows(tokenizer.backend_tokenizer, texts), batch["spans"], strict=True
    ):
        spans = spans_from_records(span_records)
        for window in windows:
            input_ids.append(window.input_ids)
            labels.append(token_label_ids(spans, window.offsets))
    return {
        "input_ids": input_ids,
        "attention_mask": [[1] * len(ids) for ids in input_ids],
        "labels": labels,
        LENGTH_COLUMN: [len(ids) for ids in input_ids],
    }


def windowed_split(
    data_dir: Path, split: str, tokenizer: PreTrainedTokenizerFast
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
    metrics: dict[str, float] = {}
    for name, scores in strict_entity_scores(gold, predicted).items():
        prefix = "overall" if name == OVERALL_REPORT_KEY else name
        metrics[f"{prefix}_precision"] = scores.precision
        metrics[f"{prefix}_recall"] = scores.recall
        metrics[f"{prefix}_f1"] = scores.f1
    return metrics


def training_arguments(
    config: TrainingConfig, options: RunOptions
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(options.output_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        max_steps=options.max_steps or -1,
        per_device_train_batch_size=config.device_batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_share,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=CHECKPOINTS_KEPT,
        logging_steps=options.logging_steps,
        train_sampling_strategy=SAMPLING_STRATEGY,
        length_column_name=LENGTH_COLUMN,
        disable_tqdm=True,
        torch_empty_cache_steps=EMPTY_CACHE_EVERY_STEPS,
        seed=TRAINING_SEED,
        data_seed=TRAINING_SEED,
        dataloader_pin_memory=False,
        use_cpu=options.use_cpu,
        report_to="none",
    )


class TrainingDivergedError(RuntimeError):
    pass


GUARDED_LOG_KEYS = ("loss", "grad_norm")


class NonFiniteLossGuard(TrainerCallback):
    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        for key in GUARDED_LOG_KEYS:
            value = (logs or {}).get(key)
            if value is not None and not math.isfinite(float(value)):
                raise TrainingDivergedError(
                    f"{key} became {value} at step {state.global_step}; stopping"
                )


def accelerator_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"{chip_name()} GPU through MPS"
    return f"{chip_name()} CPU"


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    datasets: SplitDatasets,
    config: TrainingConfig,
    options: RunOptions,
) -> Trainer:
    trainer = Trainer(
        model=model,
        args=training_arguments(config, options),
        train_dataset=datasets.train,
        eval_dataset=datasets.validation,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_loss_func=weighted_token_loss(class_weights(datasets.train["labels"])),
        compute_metrics=entity_metrics,
        preprocess_logits_for_metrics=most_likely_labels,
        callbacks=[NonFiniteLossGuard()],
    )
    trainer.loss_is_scaled_for_ga = False
    return trainer


def load_base_model(
    config: TrainingConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
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


class StepTimer(TrainerCallback):
    def __init__(self) -> None:
        self.first_step: tuple[int, float] | None = None
        self.last_step: tuple[int, float] | None = None

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        now = (state.global_step, time.perf_counter())
        if self.first_step is None:
            self.first_step = now
        self.last_step = now

    def steps_per_second(self) -> float | None:
        if self.first_step is None or self.last_step is None:
            return None
        steps = self.last_step[0] - self.first_step[0]
        seconds = self.last_step[1] - self.first_step[1]
        return steps / seconds if seconds > 0 else None


def loss_history(state: TrainerState) -> list[tuple[int, float]]:
    return [
        (int(entry["step"]), float(entry["loss"]))
        for entry in state.log_history
        if "loss" in entry
    ]


def last_evaluation(trainer: Trainer) -> dict[str, float]:
    evaluations = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
    if evaluations:
        return evaluations[-1]
    return trainer.evaluate()


def planned_steps(train_windows: int, config: TrainingConfig) -> int:
    return config.epochs * -(-train_windows // config.effective_batch_size)


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    preset: str
    base_model: str
    base_model_revision: str
    dataset_revision: str
    training_seed: int
    config: dict[str, Any]
    device: str
    accelerator: str
    environment: dict[str, Any]
    train_windows: int
    validation_windows: int
    planned_steps: int
    completed_steps: int
    class_weights: list[float]
    training_seconds: float
    train_steps_per_second: float | None
    loss_history: list[tuple[int, float]]
    validation_metrics: dict[str, float]


def train_detector(
    data_dir: Path,
    models_dir: Path,
    preset: ModelPreset,
    options: RunOptions | None = None,
) -> TrainingRecord:
    output_dir = models_dir / preset.value
    run_options = options or RunOptions(output_dir=output_dir / "checkpoints")
    config = PRESETS[preset]
    if run_options.device_batch_size is not None:
        config = with_device_batch(config, run_options.device_batch_size)
    model, tokenizer = load_base_model(config)
    datasets = SplitDatasets(
        train=windowed_split(data_dir, "train", tokenizer),
        validation=windowed_split(data_dir, "validation", tokenizer),
    )
    trainer = build_trainer(model, tokenizer, datasets, config, run_options)
    timer = StepTimer()
    trainer.add_callback(timer)
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=run_options.resume or None)
    elapsed = time.perf_counter() - started
    trainer.save_model(str(output_dir))
    weights = class_weights(datasets.train["labels"])
    record = TrainingRecord(
        preset=preset.value,
        base_model=config.base_model,
        base_model_revision=config.revision,
        dataset_revision=OPENPII_REVISION,
        training_seed=TRAINING_SEED,
        config=asdict(config),
        device=str(trainer.args.device),
        accelerator=accelerator_name(trainer.args.device),
        environment=environment_record(),
        train_windows=len(datasets.train),
        validation_windows=len(datasets.validation),
        planned_steps=planned_steps(len(datasets.train), config),
        completed_steps=trainer.state.global_step,
        class_weights=weights,
        training_seconds=round(elapsed, 1),
        train_steps_per_second=timer.steps_per_second(),
        loss_history=loss_history(trainer.state),
        validation_metrics=last_evaluation(trainer),
    )
    (output_dir / RUN_RECORD_NAME).write_text(json.dumps(asdict(record), indent=2))
    return record
