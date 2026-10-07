import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
TINY_WORDS = (
    *(f"w{index}" for index in range(40)),
    "Ada",
    "Quill",
    "mail",
    "ada@example.com",
    "now",
    "My",
    "password",
    "is",
)


@pytest.fixture(scope="session")
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS + TINY_WORDS)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
