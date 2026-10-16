from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForTokenClassification


class LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def load_logits_model(model_dir: Path) -> LogitsOnly:
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return LogitsOnly(model).eval()
