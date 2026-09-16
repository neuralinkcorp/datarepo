"""Run with python -I from outside the checkout after installing a built wheel."""

import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

import polars as pl
import datarepo
from datarepo.core import Filter, ParquetTable


origin = Path(datarepo.__file__).resolve()
if "site-packages" not in origin.parts:
    raise AssertionError(f"Expected a wheel installation, imported {origin}")
with TemporaryDirectory(prefix="datarepo-wheel-smoke-") as directory:
    path = Path(directory) / "example.parquet"
    pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}).write_parquet(path)
    table = ParquetTable(name="example", uri=str(path), partitioning=[])
    result = (
        table(filters=[Filter("value", ">", 10)], columns=["id", "value"])
        .collect()
        .sort("id")
    )
    assert result.to_dicts() == [{"id": 2, "value": 20}, {"id": 3, "value": 30}]
    assert result.schema == {"id": pl.Int64, "value": pl.Int64}
print(
    json.dumps(
        {
            "python": platform.python_version(),
            "isolated": bool(sys.flags.isolated),
            "imported_from": str(origin),
            "rows": result.to_dicts(),
        },
        indent=2,
    )
)
