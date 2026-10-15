import json
from pathlib import Path

from edgeveil.bench.tables import accuracy_table, render_accuracy_tables

ESTIMATE = {"value": 0.9, "low": 0.85, "high": 0.95}
PAYLOAD = {
    "base_model": "distilbert/distilbert-base-cased",
    "precision": "fp32",
    "backend": "torch",
    "test_set": {"snippets": 235},
    "bootstrap": {"resamples": 1000, "unit": "snippet"},
    "scores": {
        "overall": {
            "precision": ESTIMATE,
            "recall": ESTIMATE,
            "f1": ESTIMATE,
            "support": 377,
        },
        "EMAIL": {
            "precision": ESTIMATE,
            "recall": ESTIMATE,
            "f1": ESTIMATE,
            "support": 45,
        },
    },
}


def test_table_lists_types_in_enum_order_with_overall_last() -> None:
    table = accuracy_table(PAYLOAD, "bench/results/accuracy/x.json")
    rows = [
        line
        for line in table.splitlines()
        if line.startswith("| ") and "Type" not in line
    ]
    assert [row.split(" | ")[0] for row in rows] == ["| EMAIL", "| overall"]
    assert "0.900 [0.850, 0.950]" in table
    assert "bench/results/accuracy/x.json" in table


def test_render_writes_one_table_per_results_file(tmp_path: Path) -> None:
    results = tmp_path / "bench" / "results"
    (results / "accuracy").mkdir(parents=True)
    (results / "accuracy" / "distilbert-fp32-torch.json").write_text(
        json.dumps(PAYLOAD)
    )
    [written] = render_accuracy_tables(results)
    assert (
        written
        == results
        / "tables"
        / "accuracy-distilbert-fp32-torch.json".replace(".json", ".md")
    )
    assert "| EMAIL | 45 |" in written.read_text()
