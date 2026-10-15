import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from edgeveil.spans import EntityType

ACCURACY_DIR_NAME = "accuracy"
TABLES_DIR_NAME = "tables"
ROW_ORDER = (*(entity.value for entity in EntityType), "overall")
HEADER = "| Type | Support | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) |"
DIVIDER = "|---|---|---|---|---|"


def format_estimate(estimate: Mapping[str, float]) -> str:
    return f"{estimate['value']:.3f} [{estimate['low']:.3f}, {estimate['high']:.3f}]"


def accuracy_row(name: str, scores: Mapping[str, Any]) -> str:
    cells = [
        name,
        str(scores["support"]),
        format_estimate(scores["precision"]),
        format_estimate(scores["recall"]),
        format_estimate(scores["f1"]),
    ]
    return "| " + " | ".join(cells) + " |"


def accuracy_table(payload: Mapping[str, Any], source: str) -> str:
    test_set = payload["test_set"]
    bootstrap = payload["bootstrap"]
    resamples, unit = bootstrap["resamples"], bootstrap["unit"]
    caption = (
        f"{payload['base_model']}, {payload['precision']}, {payload['backend']}. "
        f"Test set: {test_set['snippets']} snippets. "
        f"Intervals: {resamples} bootstrap resamples by {unit}. "
        f"Source: `{source}`."
    )
    scores = payload["scores"]
    rows = [accuracy_row(name, scores[name]) for name in ROW_ORDER if name in scores]
    return "\n".join([caption, "", HEADER, DIVIDER, *rows]) + "\n"


def render_accuracy_tables(results_dir: Path) -> list[Path]:
    tables_dir = results_dir / TABLES_DIR_NAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source in sorted((results_dir / ACCURACY_DIR_NAME).glob("*.json")):
        payload = json.loads(source.read_text())
        table_path = tables_dir / f"accuracy-{source.stem}.md"
        table_path.write_text(
            accuracy_table(payload, source.relative_to(results_dir.parent).as_posix())
        )
        written.append(table_path)
    return written
