from dataclasses import dataclass, replace
from enum import StrEnum


class InvalidBatchLayoutError(ValueError):
    pass


class ModelPreset(StrEnum):
    DISTILBERT = "distilbert"
    MODERNBERT = "modernbert"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    base_model: str
    revision: str
    learning_rate: float = 5e-5
    epochs: int = 2
    device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_share: float = 0.06
    weight_decay: float = 0.01

    @property
    def effective_batch_size(self) -> int:
        return self.device_batch_size * self.gradient_accumulation_steps


PRESETS = {
    ModelPreset.DISTILBERT: TrainingConfig(
        base_model="distilbert/distilbert-base-cased",
        revision="6ea81172465e8b0ad3fddeed32b986cdcdcffcf0",
    ),
    ModelPreset.MODERNBERT: TrainingConfig(
        base_model="answerdotai/ModernBERT-base",
        revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
    ),
}


def with_device_batch(config: TrainingConfig, device_batch_size: int) -> TrainingConfig:
    effective = config.effective_batch_size
    if device_batch_size <= 0 or effective % device_batch_size:
        raise InvalidBatchLayoutError(
            f"device batch {device_batch_size} must divide effective batch {effective}"
        )
    return replace(
        config,
        device_batch_size=device_batch_size,
        gradient_accumulation_steps=effective // device_batch_size,
    )
