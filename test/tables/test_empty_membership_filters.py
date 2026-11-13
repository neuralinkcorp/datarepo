from collections import Counter
from copy import deepcopy
from pathlib import Path

import boto3
from deltalake import write_deltalake
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datarepo.core.tables.deltalake_table import DeltalakeTable
from datarepo.core.tables.filters import Filter, FilterOperator
from datarepo.core.tables.parquet_table import ParquetTable


FIELDS: list[tuple[str, pa.DataType]] = [
    ("id", pa.int64()),
    ("label", pa.string()),
    ("tags", pa.list_(pa.string())),
]
SCHEMA = pa.schema(FIELDS)
ROWS: list[dict[str, object]] = [
    {"id": 1, "label": "a", "tags": ["a"]},
    {"id": 2, "label": "b", "tags": []},
    {"id": None, "label": "n", "tags": None},
    {"id": 1, "label": "c", "tags": ["a", "b"]},
]
ALL_ROWS: list[tuple[int | None, str]] = [(1, "a"), (2, "b"), (None, "n"), (1, "c")]


@pytest.fixture(params=["delta", "parquet"])
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def local_session() -> boto3.Session:
    return boto3.Session(
        aws_access_key_id="synthetic-local-only",
        aws_secret_access_key="synthetic-local-only",
        region_name="us-east-1",
    )


def table_at(
    path: Path, backend: str, *, empty: bool = False
) -> DeltalakeTable | ParquetTable:
    data = pa.Table.from_pylist([] if empty else ROWS, schema=SCHEMA)
    location = path / "table"
    if backend == "delta":
        write_deltalake(str(location), data)
        return DeltalakeTable(name="fixture", uri=str(location), schema=SCHEMA)
    location.mkdir()
    pq.write_table(data, location / "df.parquet")
    schema: dict[str, pl.DataType] = {
        "id": pl.Int64(),
        "label": pl.String(),
        "tags": pl.List(pl.String()),
    }
    return ParquetTable(
        name="fixture", uri=str(location), partitioning=[], schema=schema
    )


def assert_rows(
    frame: pl.DataFrame,
    expected: list[tuple[int | None, str]],
    columns: tuple[str, ...] = ("id", "label"),
) -> None:
    assert frame.columns == list(columns)
    assert frame.schema["id"] == pl.Int64
    assert frame.schema["label"] == pl.String
    assert Counter(frame.select("id", "label").iter_rows()) == Counter(expected)


@pytest.mark.parametrize("values", [[], ()], ids=["list", "tuple"])
@pytest.mark.parametrize(
    ("operator", "column", "expected"),
    [
        ("in", "id", []),
        ("not in", "id", ALL_ROWS),
        ("includes any", "tags", []),
        ("includes all", "tags", ALL_ROWS),
    ],
)
def test_empty_membership_uses_empty_set_identities(
    tmp_path: Path,
    local_session: boto3.Session,
    backend: str,
    operator: FilterOperator,
    column: str,
    expected: list[tuple[int | None, str]],
    values: list[object] | tuple[object, ...],
) -> None:
    table = table_at(tmp_path, backend)
    frame = table(
        filters=[Filter(column, operator, values)],
        columns=["id", "label"],
        boto3_session=local_session,
    ).collect()
    assert_rows(frame, expected)


@pytest.mark.parametrize(
    ("filter", "expected"),
    [
        (Filter("id", "in", [1]), [(1, "a"), (1, "c")]),
        (Filter("id", "not in", [1]), [(2, "b")]),
        (Filter("tags", "includes any", ["a"]), [(1, "a"), (1, "c")]),
        (Filter("tags", "includes all", ["a", "b"]), [(1, "c")]),
    ],
)
def test_populated_membership_keeps_values_nulls_and_multiplicity(
    tmp_path: Path,
    local_session: boto3.Session,
    backend: str,
    filter: Filter,
    expected: list[tuple[int | None, str]],
) -> None:
    table = table_at(tmp_path, backend)
    assert_rows(
        table(
            filters=[filter], columns=["id", "label"], boto3_session=local_session
        ).collect(),
        expected,
    )


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ([Filter("id", "=", 1), Filter("tags", "includes any", [])], []),
        (
            [[Filter("id", "in", []), Filter("id", "=", 1)], [Filter("id", "=", 2)]],
            [(2, "b")],
        ),
        ([[Filter("id", "=", 1)], [Filter("id", "not in", [])]], ALL_ROWS),
        (
            [Filter("id", "=", 1), Filter("tags", "includes all", [])],
            [(1, "a"), (1, "c")],
        ),
    ],
    ids=["false-and", "false-or", "true-or", "true-and"],
)
def test_empty_membership_composes_without_changing_inputs(
    tmp_path: Path,
    local_session: boto3.Session,
    backend: str,
    filters: list[Filter] | list[list[Filter]],
    expected: list[tuple[int | None, str]],
) -> None:
    table = table_at(tmp_path, backend)
    before = deepcopy(filters)
    frame = table(
        filters=filters, columns=["label", "id"], boto3_session=local_session
    ).collect()
    assert_rows(frame, expected, ("label", "id"))
    assert filters == before


@pytest.mark.parametrize(
    "filters", [None, [], [[]]], ids=["none", "empty", "empty-and"]
)
def test_empty_filter_container_still_means_unfiltered(
    tmp_path: Path,
    local_session: boto3.Session,
    backend: str,
    filters: list[list[Filter]] | None,
) -> None:
    table = table_at(tmp_path, backend)
    assert_rows(
        table(
            filters=filters, columns=["id", "label"], boto3_session=local_session
        ).collect(),
        ALL_ROWS,
    )


@pytest.mark.parametrize(
    "filter",
    [
        Filter("id", "in", []),
        Filter("id", "not in", []),
        Filter("tags", "includes any", []),
        Filter("tags", "includes all", []),
    ],
)
def test_empty_dataset_preserves_projected_schema(
    tmp_path: Path, local_session: boto3.Session, backend: str, filter: Filter
) -> None:
    table = table_at(tmp_path, backend, empty=True)
    assert_rows(
        table(
            filters=[filter], columns=["label", "id"], boto3_session=local_session
        ).collect(),
        [],
        ("label", "id"),
    )


@pytest.mark.parametrize("operator", ["in", "includes any"])
def test_empty_membership_still_validates_column(
    tmp_path: Path,
    local_session: boto3.Session,
    backend: str,
    operator: FilterOperator,
) -> None:
    table = table_at(tmp_path, backend)
    error = ValueError if backend == "delta" else pl.exceptions.ColumnNotFoundError
    with pytest.raises(error, match="missing"):
        table(
            filters=[Filter("missing", operator, [])],
            columns=["id", "label"],
            boto3_session=local_session,
        ).collect()


@pytest.mark.parametrize("operator", ["includes any", "includes all"])
def test_empty_array_membership_still_requires_array_column(
    tmp_path: Path,
    local_session: boto3.Session,
    backend: str,
    operator: FilterOperator,
) -> None:
    table = table_at(tmp_path, backend)
    error = AssertionError if backend == "delta" else pl.exceptions.SchemaError
    with pytest.raises(error):
        table(
            filters=[Filter("id", operator, [])],
            columns=["id", "label"],
            boto3_session=local_session,
        ).collect()
