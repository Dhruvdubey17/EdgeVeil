from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from edgeveil.dataset.prepare import PreparationReport

DEFAULT_DATA_DIR = Path("data")

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
    console.print(count_table("Rows", report_rows(report)))
    console.print(
        count_table(
            "Entities in rows dropped for offset mismatch",
            report.mismatched_entities_by_label,
        )
    )
    console.print(
        count_table("Raw rows by source dataset", report.raw_rows_by_source_dataset)
    )
    for split, counts in report.spans_by_type.items():
        console.print(count_table(f"Spans by type ({split})", counts))


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
    rows["train: dropped, text also in validation"] = (
        report.train_dropped_as_in_validation
    )
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
