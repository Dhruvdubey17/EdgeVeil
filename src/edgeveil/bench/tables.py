import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from edgeveil.evaluation.detection import Detection, results_stem
from edgeveil.spans import EntityType

ACCURACY_DIR_NAME = "accuracy"
TABLES_DIR_NAME = "tables"
OVERALL_ROW = "overall"
ROW_ORDER = (*(entity.value for entity in EntityType), OVERALL_ROW)
DETECTION_NAMES = {
    Detection.MODEL: "model only",
    Detection.VALIDATORS: "validators only",
    Detection.COMBINED: "model and validators",
}
STRICT_MEASURES = ("precision", "recall", "f1")
SECONDARY_MEASURES = ("overlap_precision", "overlap_recall", "char_recall")
ACCURACY_COLUMNS = (
    "Type",
    "Support",
    "Precision (95% CI)",
    "Recall (95% CI)",
    "F1 (95% CI)",
    "Overlap precision",
    "Overlap recall",
    "Char recall",
)
ERROR_COLUMNS = (
    "Type",
    "Missed",
    "Boundary miss",
    "Wrong-type miss",
    "Spurious FP",
    "Boundary FP",
    "Wrong-type FP",
)
MISS_KINDS = ("missed", "boundary", "wrong_type")
FALSE_POSITIVE_KINDS = ("spurious", "boundary", "wrong_type")
ABLATION_MODES = (Detection.MODEL, Detection.VALIDATORS, Detection.COMBINED)


def format_estimate(estimate: Mapping[str, float]) -> str:
    return f"{estimate['value']:.3f} [{estimate['low']:.3f}, {estimate['high']:.3f}]"


def markdown_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines)


def detection_of(payload: Mapping[str, Any]) -> Detection:
    return Detection(payload["detection"])


def accuracy_row(name: str, payload: Mapping[str, Any]) -> list[str]:
    strict = payload["scores"][name]
    secondary = payload["secondary_scores"][name]
    return [
        name,
        str(strict["support"]),
        *(format_estimate(strict[measure]) for measure in STRICT_MEASURES),
        *(format_estimate(secondary[measure]) for measure in SECONDARY_MEASURES),
    ]


def error_row(name: str, errors: Mapping[str, Any]) -> list[str]:
    misses = errors["misses"][name]
    false_positives = errors["false_positives"][name]
    return [
        name,
        *(str(misses[kind]) for kind in MISS_KINDS),
        *(str(false_positives[kind]) for kind in FALSE_POSITIVE_KINDS),
    ]


def error_total_row(errors: Mapping[str, Any]) -> list[str]:
    misses = errors["misses"].values()
    false_positives = errors["false_positives"].values()
    return [
        OVERALL_ROW,
        *(str(sum(row[kind] for row in misses)) for kind in MISS_KINDS),
        *(
            str(sum(row[kind] for row in false_positives))
            for kind in FALSE_POSITIVE_KINDS
        ),
    ]


def caption(payload: Mapping[str, Any], sources: str) -> str:
    test_set = payload["test_set"]
    bootstrap = payload["bootstrap"]
    resamples, unit = bootstrap["resamples"], bootstrap["unit"]
    return (
        f"{payload['base_model']}, {payload['precision']}, {payload['backend']}. "
        f"Test set: {test_set['snippets']} snippets. "
        f"Intervals: {resamples} bootstrap resamples by {unit}. "
        f"Source: {sources}."
    )


def accuracy_table(payload: Mapping[str, Any], source: str) -> str:
    detection = DETECTION_NAMES[detection_of(payload)]
    scores = payload["scores"]
    rows = [accuracy_row(name, payload) for name in ROW_ORDER if name in scores]
    errors = payload["errors"]
    error_rows = [error_row(entity.value, errors) for entity in EntityType]
    sections = [
        f"Detection: {detection}. {caption(payload, f'`{source}`')}",
        "Strict precision, recall, and F1 are the headline. Overlap scores count "
        "a span as matched if it overlaps a span of the same type. Char recall is "
        "the share of gold span characters covered by any emitted span.",
        markdown_table(ACCURACY_COLUMNS, rows),
        "Strict errors. A miss is a gold span without an exact match; a false "
        "positive (FP) is an emitted span without one. Boundary: overlaps a span "
        "of the same type. Wrong type: overlaps only spans of other types.",
        markdown_table(ERROR_COLUMNS, [*error_rows, error_total_row(errors)]),
    ]
    return "\n\n".join(sections) + "\n"


def ablation_row(
    name: str, payloads: Mapping[Detection, Mapping[str, Any]]
) -> list[str]:
    support = payloads[Detection.MODEL]["scores"][name]["support"]
    cells = [name, str(support)]
    for mode in ABLATION_MODES:
        scores = payloads[mode]["scores"][name]
        cells += [format_estimate(scores[measure]) for measure in ("recall", "f1")]
    return cells


def ablation_table(
    payloads: Mapping[Detection, Mapping[str, Any]], sources: Sequence[str]
) -> str:
    columns = ["Type", "Support"]
    for mode in ABLATION_MODES:
        columns += [f"Recall, {DETECTION_NAMES[mode]}", f"F1, {DETECTION_NAMES[mode]}"]
    gold_types = payloads[Detection.MODEL]["scores"]
    rows = [ablation_row(name, payloads) for name in ROW_ORDER if name in gold_types]
    listed = ", ".join(f"`{source}`" for source in sources)
    return (
        "\n\n".join(
            [
                "Validator ablation, strict scores. "
                + caption(payloads[Detection.MODEL], listed),
                markdown_table(columns, rows),
            ]
        )
        + "\n"
    )


def ablation_sources(
    combined_path: Path, payload: Mapping[str, Any]
) -> dict[Detection, Path]:
    run_name = combined_path.stem.removesuffix(f"-{Detection.COMBINED.value}")
    directory = combined_path.parent
    return {
        mode: directory / f"{results_stem(mode, payload['preset'], run_name)}.json"
        for mode in ABLATION_MODES
    }


def render_ablation_tables(results_dir: Path, tables_dir: Path) -> list[Path]:
    written: list[Path] = []
    pattern = f"*-{Detection.COMBINED.value}.json"
    for combined in sorted((results_dir / ACCURACY_DIR_NAME).glob(pattern)):
        sources = ablation_sources(combined, json.loads(combined.read_text()))
        if not all(path.exists() for path in sources.values()):
            continue
        payloads = {
            mode: json.loads(path.read_text()) for mode, path in sources.items()
        }
        run_name = sources[Detection.MODEL].stem
        table_path = tables_dir / f"ablation-{run_name}.md"
        names = [path.as_posix() for path in sources.values()]
        table_path.write_text(ablation_table(payloads, names))
        written.append(table_path)
    return written


def render_accuracy_tables(results_dir: Path) -> list[Path]:
    tables_dir = results_dir / TABLES_DIR_NAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source in sorted((results_dir / ACCURACY_DIR_NAME).glob("*.json")):
        payload = json.loads(source.read_text())
        table_path = tables_dir / f"accuracy-{source.stem}.md"
        table_path.write_text(accuracy_table(payload, source.as_posix()))
        written.append(table_path)
    return written + render_ablation_tables(results_dir, tables_dir)
