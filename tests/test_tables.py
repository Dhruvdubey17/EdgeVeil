import json
from pathlib import Path
from typing import Any

from edgeveil.bench.tables import accuracy_table, render_accuracy_tables
from edgeveil.spans import EntityType

ESTIMATE = {"value": 0.9, "low": 0.85, "high": 0.95}
SECONDARY = {
    "overlap_precision": ESTIMATE,
    "overlap_recall": ESTIMATE,
    "char_recall": {"value": 0.97, "low": 0.95, "high": 0.99},
}


def strict(support: int) -> dict[str, Any]:
    return {
        "precision": ESTIMATE,
        "recall": ESTIMATE,
        "f1": ESTIMATE,
        "support": support,
    }


def payload(detection: str) -> dict[str, Any]:
    misses = {"missed": 0, "boundary": 0, "wrong_type": 0}
    false_positives = {"spurious": 0, "boundary": 0, "wrong_type": 0}
    errors = {
        "misses": {entity.value: dict(misses) for entity in EntityType},
        "false_positives": {
            entity.value: dict(false_positives) for entity in EntityType
        },
    }
    errors["misses"]["EMAIL"]["boundary"] = 3
    return {
        "detection": detection,
        "preset": "distilbert",
        "base_model": "distilbert/distilbert-base-cased",
        "precision": "fp32",
        "backend": "torch",
        "test_set": {"snippets": 235},
        "bootstrap": {"resamples": 1000, "unit": "snippet"},
        "scores": {"overall": strict(377), "EMAIL": strict(45)},
        "secondary_scores": {"overall": SECONDARY, "EMAIL": SECONDARY},
        "errors": errors,
    }


def table_rows(table: str) -> list[str]:
    return [line for line in table.splitlines() if line.startswith("| EMAIL")]


def test_table_lists_types_in_enum_order_with_overall_last() -> None:
    table = accuracy_table(payload("model"), "bench/results/accuracy/x.json")
    accuracy_rows = table.split("\n\n")[2].splitlines()[2:]
    assert [row.split(" | ")[0] for row in accuracy_rows] == ["| EMAIL", "| overall"]
    assert "0.900 [0.850, 0.950]" in table
    assert "0.970 [0.950, 0.990]" in table
    assert "bench/results/accuracy/x.json" in table
    assert "Detection: model only." in table


def test_error_counts_follow_the_accuracy_table() -> None:
    table = accuracy_table(payload("model"), "x.json")
    assert table_rows(table)[1] == "| EMAIL | 0 | 3 | 0 | 0 | 0 | 0 |"
    assert "| overall | 0 | 3 | 0 | 0 | 0 | 0 |" in table


def test_render_writes_one_table_per_results_file_and_an_ablation(
    tmp_path: Path,
) -> None:
    results = tmp_path / "bench" / "results"
    (results / "accuracy").mkdir(parents=True)
    names = {
        "model": "distilbert-fp32-torch",
        "validators": "distilbert-validators",
        "combined": "distilbert-fp32-torch-combined",
    }
    for detection, stem in names.items():
        (results / "accuracy" / f"{stem}.json").write_text(
            json.dumps(payload(detection))
        )
    written = render_accuracy_tables(results)
    assert [path.name for path in written] == [
        "accuracy-distilbert-fp32-torch-combined.md",
        "accuracy-distilbert-fp32-torch.md",
        "accuracy-distilbert-validators.md",
        "ablation-distilbert-fp32-torch.md",
    ]
    ablation = written[-1].read_text()
    assert "F1, model and validators" in ablation
    assert table_rows(ablation)[0].count("0.900 [0.850, 0.950]") == 6


def test_ablation_needs_all_three_results(tmp_path: Path) -> None:
    results = tmp_path / "bench" / "results"
    (results / "accuracy").mkdir(parents=True)
    (results / "accuracy" / "distilbert-fp32-torch-combined.json").write_text(
        json.dumps(payload("combined"))
    )
    written = render_accuracy_tables(results)
    assert [path.name for path in written] == [
        "accuracy-distilbert-fp32-torch-combined.md"
    ]
