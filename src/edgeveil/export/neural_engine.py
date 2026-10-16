from pathlib import Path

import torch
from torch import nn
from torch.nn import functional
from transformers import (
    AutoModelForTokenClassification,
    DistilBertForTokenClassification,
)
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertSelfAttention,
    TransformerBlock,
)

CHANNEL_AXIS = 1
KEY_AXIS = 1
DISTILBERT_LAYER_NORM_EPSILON = 1e-12


class UnsupportedArchitectureError(TypeError):
    pass


def conv_from_linear(linear: nn.Linear) -> nn.Conv2d:
    conv = nn.Conv2d(linear.in_features, linear.out_features, kernel_size=1)
    conv.weight = nn.Parameter(linear.weight.detach()[:, :, None, None].clone())
    conv.bias = nn.Parameter(linear.bias.detach().clone())
    return conv


class ChannelLayerNorm(nn.Module):
    def __init__(self, layer_norm: nn.LayerNorm) -> None:
        super().__init__()
        self.epsilon = layer_norm.eps
        self.weight = nn.Parameter(layer_norm.weight.detach()[None, :, None, None])
        self.bias = nn.Parameter(layer_norm.bias.detach()[None, :, None, None])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        centered = hidden - hidden.mean(dim=CHANNEL_AXIS, keepdim=True)
        variance = (centered * centered).mean(dim=CHANNEL_AXIS, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.epsilon)
        return normalized * self.weight + self.bias


class PerHeadSelfAttention(nn.Module):
    def __init__(self, attention: DistilBertSelfAttention, head_count: int) -> None:
        super().__init__()
        self.query = conv_from_linear(attention.q_lin)
        self.key = conv_from_linear(attention.k_lin)
        self.value = conv_from_linear(attention.v_lin)
        self.output = conv_from_linear(attention.out_lin)
        self.head_size = attention.q_lin.out_features // head_count
        self.scale = self.head_size**-0.5

    def forward(
        self, hidden: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        queries = (self.query(hidden) * self.scale).split(self.head_size, CHANNEL_AXIS)
        keys = self.key(hidden).transpose(1, 3).split(self.head_size, dim=3)
        values = self.value(hidden).split(self.head_size, CHANNEL_AXIS)
        heads = []
        for query, key, value in zip(queries, keys, values, strict=True):
            scores = torch.einsum("bchq,bkhc->bkhq", query, key) + attention_bias
            weights = scores.softmax(dim=KEY_AXIS)
            heads.append(torch.einsum("bkhq,bchk->bchq", weights, value))
        return self.output(torch.cat(heads, dim=CHANNEL_AXIS))


class NeuralEngineBlock(nn.Module):
    def __init__(self, block: TransformerBlock, head_count: int) -> None:
        super().__init__()
        self.attention = PerHeadSelfAttention(block.attention, head_count)
        self.attention_norm = ChannelLayerNorm(block.sa_layer_norm)
        self.expand = conv_from_linear(block.ffn.lin1)
        self.contract = conv_from_linear(block.ffn.lin2)
        self.output_norm = ChannelLayerNorm(block.output_layer_norm)

    def forward(
        self, hidden: torch.Tensor, attention_bias: torch.Tensor
    ) -> torch.Tensor:
        attended = self.attention_norm(self.attention(hidden, attention_bias) + hidden)
        expanded = functional.gelu(self.expand(attended))
        return self.output_norm(self.contract(expanded) + attended)


def transformer_blocks(
    model: DistilBertForTokenClassification,
) -> list[TransformerBlock]:
    blocks = list(model.distilbert.transformer.layer)
    typed = [block for block in blocks if isinstance(block, TransformerBlock)]
    if len(typed) != len(blocks):
        raise UnsupportedArchitectureError("expected DistilBERT transformer blocks")
    return typed


class NeuralEngineTokenClassifier(nn.Module):
    def __init__(self, model: DistilBertForTokenClassification) -> None:
        super().__init__()
        embeddings = model.distilbert.embeddings
        head_count = model.config.n_heads
        self.word_embeddings = embeddings.word_embeddings
        self.position_embeddings = embeddings.position_embeddings
        self.embedding_norm = ChannelLayerNorm(embeddings.LayerNorm)
        self.blocks = nn.ModuleList(
            NeuralEngineBlock(block, head_count) for block in transformer_blocks(model)
        )
        self.classifier = conv_from_linear(model.classifier)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.word_embeddings(input_ids) + self.position_embeddings(
            position_ids
        )
        hidden = self.embedding_norm(embedded.transpose(1, 2).unsqueeze(2))
        for block in self.blocks:
            hidden = block(hidden, attention_bias)
        return self.classifier(hidden)


def load_neural_engine_model(model_dir: Path) -> NeuralEngineTokenClassifier:
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    if not isinstance(model, DistilBertForTokenClassification):
        raise UnsupportedArchitectureError(
            f"the Neural Engine layout supports DistilBERT, got {type(model).__name__}"
        )
    return NeuralEngineTokenClassifier(model.eval()).eval()
