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
    error_example_lines,
    evaluate_test_set,
    load_thresholds,
    write_accuracy_results,
)
from edgeveil.evaluation.detection import Detection
from edgeveil.evaluation.errors import error_counts
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
    reports = evaluate_test_set(
        model_dir, test_set, thresholds, resamples=50, device=torch.device("cpu")
    )
    assert set(reports) == set(Detection)
    report = reports[Detection.MODEL]
    assert report.run.snippets == 3
    assert {"PERSON", "EMAIL", "overall"} <= set(report.scores)
    assert report.scores["PERSON"].support == 3
    path = tmp_path / "bench" / "results" / "accuracy.json"
    write_accuracy_results(report, path)
    saved = path.read_text()
    assert "example.com" not in saved
    assert "Ada" not in saved
    payload = json.loads(saved)
    assert payload["detection"] == "model"
    assert payload["bootstrap"]["unit"] == "snippet"
    assert set(payload["scores"]["PERSON"]) == {"precision", "recall", "f1", "support"}
    assert set(payload["secondary_scores"]["overall"]) == {
        "overlap_precision",
        "overlap_recall",
        "char_recall",
    }
    assert len(payload["test_set"]["sha256"]) == 64
    assert len(payload["model_sha256"]) == 64
    assert len(payload["thresholds_sha256"]) == 64


def test_validators_alone_find_the_email_and_miss_the_name(
    tiny_setup: tuple[Path, Path, Path],
) -> None:
    model_dir, test_set, thresholds = tiny_setup
    reports = evaluate_test_set(
        model_dir, test_set, thresholds, resamples=20, device=torch.device("cpu")
    )
    validators = reports[Detection.VALIDATORS]
    assert validators.scores["EMAIL"].recall.value == 1.0
    assert validators.secondary["EMAIL"].char_recall.value == 1.0
    counts = error_counts(validators.errors)
    assert counts["misses"]["PERSON"] == {"missed": 3, "boundary": 0, "wrong_type": 0}
    assert counts["false_positives"]["EMAIL"]["spurious"] == 0


def test_error_examples_show_the_span_in_its_context(
    tiny_setup: tuple[Path, Path, Path],
) -> None:
    model_dir, test_set, thresholds = tiny_setup
    reports = evaluate_test_set(
        model_dir, test_set, thresholds, resamples=20, device=torch.device("cpu")
    )
    lines = error_example_lines(reports[Detection.VALIDATORS])
    assert lines[0] == (
        "misses PERSON missed (snippet 1): [Ada Quill] mail ada@example.com now"
    )
