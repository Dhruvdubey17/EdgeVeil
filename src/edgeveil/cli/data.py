from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from edgeveil.dataset.credentials import AUGMENTATION_SEED

if TYPE_CHECKING:
    from edgeveil.dataset.prepare import PreparationReport

DEFAULT_DATA_DIR = Path("data")
DEFAULT_TEST_SET_PATH = DEFAULT_DATA_DIR / "test" / "test_set.jsonl"
LABELING_ISSUES_EXIT_CODE = 1

data_app = typer.Typer(help="Download and prepare training data.", no_args_is_help=True)
console = Console()

DataDir = Annotated[
    Path, typer.Option(help="Directory for downloaded and prepared data.")
]


@data_app.command(
    help="Stream the English rows of OpenPII 1.5M into parquet under data/."
)
def download(data_dir: DataDir = DEFAULT_DATA_DIR) -> None:
    from edgeveil.dataset.download import download_english_subset

    record = download_english_subset(data_dir)
    console.print(f"Saved {record.language} rows of {record.repository}.")
    console.print(f"Revision {record.revision}, downloaded {record.downloaded_at}.")


@data_app.command(help="Check offsets, deduplicate, and map labels to entity types.")
def prepare(data_dir: DataDir = DEFAULT_DATA_DIR) -> None:
    from edgeveil.dataset.prepare import prepare_openpii

    report = prepare_openpii(data_dir)
    tables = {
        "Rows": report_rows(report),
        "Entities in rows dropped for offset mismatch": (
            report.mismatched_entities_by_label
        ),
        "Labels in rows dropped as off-card": report.off_card_labels,
        "Raw rows by source dataset": report.raw_rows_by_source_dataset,
    }
    for split, counts in report.spans_by_type.items():
        tables[f"Spans by type ({split})"] = counts
    for title, counts in tables.items():
        console.print(count_table(title, counts))


@data_app.command(
    help="Add seeded synthetic credentials and hard negatives under data/."
)
def augment(
    data_dir: DataDir = DEFAULT_DATA_DIR,
    seed: Annotated[
        int, typer.Option(help="Seed for the generator.")
    ] = AUGMENTATION_SEED,
) -> None:
    from edgeveil.dataset.augment import augment_openpii

    report = augment_openpii(data_dir, seed)
    console.print(f"Seed {report.seed}.")
    for split, counts in report.rows_by_origin.items():
        console.print(count_table(f"Rows by origin ({split})", counts))
    console.print(count_table("CREDENTIAL spans", report.credential_spans))
    console.print(
        count_table("Name pairs eligible for a connector", report.name_pairs_eligible)
    )
    console.print(
        count_table("Name connectors inserted", report.name_connectors_inserted)
    )
    for split, share in report.credential_row_share.items():
        console.print(f"{split}: {share:.2%} of rows contain a CREDENTIAL span.")


@data_app.command(help="Show the gaps that keep name or address parts from merging.")
def gaps(
    data_dir: DataDir = DEFAULT_DATA_DIR,
    split: Annotated[str, typer.Option(help="Split to inspect.")] = "validation",
    top: Annotated[int, typer.Option(help="Number of gaps to list per kind.")] = 20,
) -> None:
    from edgeveil.dataset.gaps import gap_report

    report = gap_report(data_dir, split, top)
    for kind, rows in report.top_split_gaps.items():
        adjacent = report.adjacent_pairs.get(kind, 0)
        split_count = report.split_pairs[kind]
        console.print(
            f"{kind}: {split_count:,} of {adjacent:,} adjacent pairs still split"
        )
        table = Table(title=f"Top split gaps ({kind}, {split})")
        for column in ("Count", "Left", "Right", "Gap"):
            table.add_column(column)
        for left, right, gap, count in rows:
            table.add_row(f"{count:,}", left, right, gap)
        console.print(table)


@data_app.command(
    name="import-labels",
    help="Convert a Label Studio JSON export into the hand-labeled test set.",
)
def import_labels(
    export_path: Annotated[
        Path,
        typer.Argument(
            help="Label Studio JSON export file.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path, typer.Option(help="Where to write the test set JSONL.")
    ] = DEFAULT_TEST_SET_PATH,
) -> None:
    from edgeveil.evaluation.label_studio import import_label_studio_export

    issues = import_label_studio_export(export_path, output)
    if not issues:
        console.print(f"Wrote the test set to {output}.")
        return
    table = Table(title="Labeling issues (nothing written)")
    for column in ("Task", "Start", "End", "Problem"):
        table.add_column(column)
    for issue in issues:
        table.add_row(
            str(issue.task_id), str(issue.start), str(issue.end), issue.problem
        )
    console.print(table)
    raise typer.Exit(code=LABELING_ISSUES_EXIT_CODE)


def report_rows(report: "PreparationReport") -> dict[str, int]:
    rows: dict[str, int] = {}
    for split, count in report.raw_rows.items():
        rows[f"{split}: raw"] = count
        rows[f"{split}: dropped, offset mismatch"] = report.rows_with_offset_mismatch[
            split
        ]
        rows[f"{split}: dropped, overlap"] = report.rows_with_overlapping_entities[
            split
        ]
        rows[f"{split}: dropped, off-card label"] = report.rows_with_off_card_labels[
            split
        ]
    rows["train: dropped, also in validation"] = report.train_dropped_as_in_validation
    rows["train: dropped, repeat"] = report.train_dropped_as_repeat
    rows["validation: dropped, repeat"] = report.validation_dropped_as_repeat
    for split, count in report.kept_rows.items():
        rows[f"{split}: kept"] = count
    return rows


def count_table(title: str, counts: dict[str, int]) -> Table:
    table = Table(title=title)
    table.add_column("Item")
    table.add_column("Count", justify="right")
    for item, count in counts.items():
        table.add_row(item, f"{count:,}")
    return table
