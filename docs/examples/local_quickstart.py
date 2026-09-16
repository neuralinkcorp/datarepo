"""Run a local Parquet query and join using synthetic data; no credentials needed."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from datarepo.core import Filter, NlkDataFrame, ParquetTable, table


@table
def suppliers() -> NlkDataFrame:
    """Synthetic suppliers, provided as a regular Python function."""
    return pl.LazyFrame(
        {"supplier_id": [10, 20], "supplier_name": ["Supplier A", "Supplier B"]}
    )


def run_example() -> pl.DataFrame:
    with TemporaryDirectory(prefix="datarepo-quickstart-") as directory:
        path = Path(directory) / "parts.parquet"
        pl.DataFrame(
            {
                "part_id": [1, 2, 3, 4],
                "part_name": ["Bolt", "Nut", "Washer", "Bracket"],
                "supplier_id": [20, 10, 20, 10],
            }
        ).write_parquet(path)
        parts = ParquetTable(name="parts", uri=str(path), partitioning=[])

        # Join on the supplier relationship, not on the unrelated part ID.
        return (
            parts(filters=[Filter("part_id", "in", [1, 2, 3])])
            .join(suppliers(), on="supplier_id")
            .select("part_id", "part_name", "supplier_id", "supplier_name")
            .sort("part_id")
            .collect()  # Read the local file before the temporary directory is removed.
        )


if __name__ == "__main__":
    print(json.dumps(run_example().to_dicts(), indent=2))
