from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from edgeveil.detector.alignment import Offset, is_special_token, token_label_ids
from edgeveil.detector.labels import ID_TO_LABEL
from edgeveil.detector.metrics import EntityScores, strict_entity_scores
from edgeveil.detector.windows import trim_offset_whitespace
from edgeveil.spans import GoldSpan, Span


def content_offsets(
    tokenizer: PreTrainedTokenizerBase, texts: Sequence[str]
) -> list[list[Offset]]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    trimmed: list[list[Offset]] = []
    for text, offsets in zip(texts, encoded["offset_mapping"], strict=True):
        text_offsets = [
            trim_offset_whitespace(text, (start, end)) for start, end in offsets
        ]
        trimmed.append(
            [offset for offset in text_offsets if not is_special_token(offset)]
        )
    return trimmed


def iob2_tags(spans: Sequence[GoldSpan | Span], offsets: Sequence[Offset]) -> list[str]:
    gold_like = [GoldSpan(span.start, span.end, span.label) for span in spans]
    return [ID_TO_LABEL[label] for label in token_label_ids(gold_like, offsets)]


def score_emitted_spans(
    gold: Sequence[Sequence[GoldSpan]],
    emitted: Sequence[Sequence[Span]],
    offsets: Sequence[Sequence[Offset]],
) -> dict[str, EntityScores]:
    gold_tags = [
        iob2_tags(spans, text) for spans, text in zip(gold, offsets, strict=True)
    ]
    emitted_tags = [
        iob2_tags(spans, text) for spans, text in zip(emitted, offsets, strict=True)
    ]
    return strict_entity_scores(gold_tags, emitted_tags)
