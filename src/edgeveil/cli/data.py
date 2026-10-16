from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from edgeveil.dataset.credentials import AUGMENTATION_SEED

if TYPE_CHECKING:
    from collections.abc import Sequence

    from edgeveil.dataset.prepare import PreparationReport
    from edgeveil.evaluation.label_studio import LabelingIssue

DEFAULT_DATA_DIR = Path("data")
DEFAULT_TEST_SET_PATH = DEFAULT_DATA_DIR / "test" / "test_set.jsonl"
DEFAULT_CORRECTIONS_PATH = DEFAULT_DATA_DIR / "test" / "corrections.jsonl"
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
    console.print(count_table("Emails seen", report.emails_seen))
    console.print(
        count_table("Email domains rewritten", report.email_domains_rewritten)
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


ExportPaths = Annotated[
    list[Path],
    typer.Argument(
        help="Label Studio JSON exports, combined into one test set.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
]
CorrectionsPath = Annotated[
    Path,
    typer.Option(
        help="Label corrections applied on top of the exports. A missing file "
        "means no corrections."
    ),
]


def print_issues(title: str, issues: "Sequence[LabelingIssue]") -> None:
    table = Table(title=title)
    for column in ("Task", "Start", "End", "Problem"):
        table.add_column(column)
    for issue in issues:
        table.add_row(
            str(issue.task_id), str(issue.start), str(issue.end), issue.problem
        )
    console.print(table)


@data_app.command(
    name="import-labels",
    help="Convert Label Studio JSON exports into the hand-labeled test set, apply "
    "label corrections, and check every span against the labeling rules.",
)
def import_labels(
    export_paths: ExportPaths,
    output: Annotated[
        Path, typer.Option(help="Where to write the test set JSONL.")
    ] = DEFAULT_TEST_SET_PATH,
    corrections: CorrectionsPath = DEFAULT_CORRECTIONS_PATH,
) -> None:
    from edgeveil.evaluation.corrections import CorrectionError
    from edgeveil.evaluation.label_studio import import_label_studio_export

    try:
        issues = import_label_studio_export(export_paths, output, corrections)
    except CorrectionError as error:
        console.print(f"Corrections failed, nothing written: {error}")
        raise typer.Exit(code=LABELING_ISSUES_EXIT_CODE) from error
    if not issues:
        console.print(f"Wrote the test set to {output}.")
        return
    print_issues("Labeling issues (nothing written)", issues)
    raise typer.Exit(code=LABELING_ISSUES_EXIT_CODE)


@data_app.command(
    name="lint-labels",
    help="Print every labeling-rule violation with the snippet text and a proposed "
    "fix. The text goes to the terminal only.",
)
def lint_labels(
    export_paths: ExportPaths,
    corrections: CorrectionsPath = DEFAULT_CORRECTIONS_PATH,
    write_proposals: Annotated[
        bool,
        typer.Option(help="Append the proposed fixes to the corrections file."),
    ] = False,
) -> None:
    from edgeveil.evaluation.corrections import (
        append_corrections,
        correction_lines,
        finding_lines,
    )
    from edgeveil.evaluation.label_studio import review_labels

    reviews, issues = review_labels(export_paths, corrections)
    if issues:
        print_issues("Labeling issues", issues)
        raise typer.Exit(code=LABELING_ISSUES_EXIT_CODE)
    proposals = [p for review in reviews for p in review.proposals]
    console.rule(f"Proposed corrections ({len(proposals)})")
    for review in reviews:
        for proposal in review.proposals:
            for line in correction_lines(review.snippet, proposal):
                console.print(line, markup=False, highlight=False)
    undecided = [(r.snippet, f) for r in reviews for f in r.undecided]
    console.rule(f"Needs a decision ({len(undecided)})")
    for snippet, finding in undecided:
        for line in finding_lines(snippet, finding):
            console.print(line, markup=False, highlight=False)
    if write_proposals:
        append_corrections(corrections, proposals)
        console.print(f"Appended {len(proposals)} corrections to {corrections}.")


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
