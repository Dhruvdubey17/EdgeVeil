import json
from pathlib import Path

import pytest
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForTokenClassification,
    PreTrainedTokenizerFast,
)

from edgeveil.detector.labels import ID_TO_LABEL, LABEL_TO_ID
from edgeveil.evaluation.accuracy import (
    UnsupportedConfigurationError,
    evaluate_test_set,
    load_thresholds,
    write_accuracy_results,
)
from edgeveil.evaluation.test_set import LabeledSnippet, write_test_set
from edgeveil.spans import EntityType, GoldSpan

TEXT = "Ada Quill mail ada@example.com now"


@pytest.fixture
def tiny_setup(
    tiny_tokenizer: PreTrainedTokenizerFast, tmp_path: Path
) -> tuple[Path, Path, Path]:
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
    model_dir = tmp_path / "models" / "distilbert"
    DistilBertForTokenClassification(config).save_pretrained(model_dir)
    tiny_tokenizer.save_pretrained(model_dir)
    test_set = tmp_path / "data" / "test" / "test_set.jsonl"
    gold = (GoldSpan(0, 9, EntityType.PERSON), GoldSpan(15, 30, EntityType.EMAIL))
    write_test_set(
        test_set, [LabeledSnippet(index, TEXT, gold) for index in range(1, 4)]
    )
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {"chosen": {entity.value: {"threshold": 0.5} for entity in EntityType}}
        )
    )
    return model_dir, test_set, thresholds


def test_thresholds_are_read_per_type(tiny_setup: tuple[Path, Path, Path]) -> None:
    _, _, thresholds = tiny_setup
    assert load_thresholds(thresholds) == dict.fromkeys(EntityType, 0.5)


def test_missing_type_in_thresholds_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"chosen": {"EMAIL": {"threshold": 0.5}}}))
    with pytest.raises(UnsupportedConfigurationError):
        load_thresholds(path)


def test_evaluation_scores_every_gold_type_and_writes_numbers_only(
    tiny_setup: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    model_dir, test_set, thresholds = tiny_setup
    report = evaluate_test_set(
        model_dir, test_set, thresholds, resamples=50, device=torch.device("cpu")
    )
    assert report.snippets == 3
    assert {"PERSON", "EMAIL", "overall"} <= set(report.scores)
    assert report.scores["PERSON"].support == 3
    path = tmp_path / "bench" / "results" / "accuracy.json"
    write_accuracy_results(report, path)
    saved = path.read_text()
    assert "example.com" not in saved
    assert "Ada" not in saved
    payload = json.loads(saved)
    assert payload["bootstrap"]["unit"] == "snippet"
    assert set(payload["scores"]["PERSON"]) == {"precision", "recall", "f1", "support"}
    assert len(payload["test_set"]["sha256"]) == 64
