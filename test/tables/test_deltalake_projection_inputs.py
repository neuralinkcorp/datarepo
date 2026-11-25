from collections import Counter
from pathlib import Path
from typing import TypeVar

import boto3
from deltalake import DeltaTable, QueryBuilder, write_deltalake
from deltalake.exceptions import DeltaError
import polars as pl
import pyarrow as pa
import pytest

import datarepo.core.tables.deltalake_table as implementation
from datarepo.core.tables.deltalake_table import DeltalakeTable


_Value = TypeVar("_Value")

SCHEMA = pa.schema([("id", pa.int64()), ("start", pa.int64()), ("end", pa.int64())])
ROWS = [{"id": 1, "start": 10, "end": 15}, {"id": 2, "start": 20, "end": 28}]


@pytest.fixture
def local_session() -> boto3.Session:
    return boto3.Session(
        aws_access_key_id="synthetic-local-only",
        aws_secret_access_key="synthetic-local-only",
        region_name="us-east-1",
    )


def table_at(
    path: Path,
    rows: list[dict[str, _Value]],
    schema: pa.Schema,
    expressions: list[tuple[pl.Expr, str]],
    *,
    unique_columns: list[str] | None = None,
    stored_schema: pa.Schema | None = None,
) -> DeltalakeTable:
    location = path / "table"
    write_deltalake(
        str(location),
        pa.Table.from_pylist(
            rows, schema=stored_schema if stored_schema is not None else schema
        ),
    )
    return DeltalakeTable(
        name="fixture",
        uri=str(location),
        schema=schema,
        extra_cols=expressions,
        unique_columns=unique_columns,
    )


def assert_rows(
    frame: pl.DataFrame, columns: list[str], expected: list[dict[str, _Value]]
) -> None:
    assert frame.columns == columns
    assert Counter(frame.iter_rows()) == Counter(
        tuple(row[column] for column in columns) for row in expected
    )


def duration() -> list[tuple[pl.Expr, str]]:
    return [((pl.col("end") - pl.col("start")).alias("duration"), "Int64")]


@pytest.mark.parametrize(
    "columns",
    [None, ["duration"], ["id", "duration"], ["id"], ["duration", "id"]],
    ids=["full", "derived-only", "id-and-derived", "id-only", "reordered"],
)
def test_projection_preserves_derived_values(
    tmp_path: Path, local_session: boto3.Session, columns: list[str] | None
) -> None:
    table = table_at(tmp_path, ROWS, SCHEMA, duration())
    expected = [
        {"id": 1, "start": 10, "end": 15, "duration": 5},
        {"id": 2, "start": 20, "end": 28, "duration": 8},
    ]
    selected = columns if columns is not None else list(expected[0])
    frame = table(columns=columns, boto3_session=local_session).collect()
    assert_rows(frame, selected, expected)


def test_physical_aliases_preserve_parallel_expression_inputs(
    tmp_path: Path, local_session: boto3.Session
) -> None:
    table = table_at(
        tmp_path,
        ROWS,
        SCHEMA,
        [
            ((pl.col("end") - pl.col("start")).alias("start"), "Int64"),
            ((pl.col("start") * 2).alias("twice"), "Int64"),
        ],
    )
    selected = ["id", "start", "twice"]
    expected = [{"id": 1, "start": 5, "twice": 20}, {"id": 2, "start": 8, "twice": 40}]
    assert_rows(
        table(boto3_session=local_session).collect().select(selected),
        selected,
        expected,
    )
    assert_rows(
        table(columns=selected, boto3_session=local_session).collect(),
        selected,
        expected,
    )


@pytest.mark.parametrize(
    ("expression", "totals"),
    [
        (pl.sum_horizontal(pl.all()).alias("total"), [26, 50]),
        (pl.sum_horizontal(pl.col("^(start|end)$")).alias("total"), [25, 48]),
    ],
    ids=["wildcard", "regex"],
)
def test_schema_sensitive_expressions_see_the_complete_declared_schema(
    tmp_path: Path,
    local_session: boto3.Session,
    expression: pl.Expr,
    totals: list[int],
) -> None:
    table = table_at(tmp_path, ROWS, SCHEMA, [(expression, "Int64")])
    expected = [{"id": 1, "total": totals[0]}, {"id": 2, "total": totals[1]}]
    selected = ["id", "total"]
    assert_rows(
        table(boto3_session=local_session).collect().select(selected),
        selected,
        expected,
    )
    assert_rows(
        table(columns=selected, boto3_session=local_session).collect(),
        selected,
        expected,
    )


def test_window_inputs_include_the_unrequested_partition_column(
    tmp_path: Path, local_session: boto3.Session
) -> None:
    rows = [dict(row, group="a") for row in ROWS]
    schema = SCHEMA.append(pa.field("group", pa.string()))
    table = table_at(
        tmp_path,
        rows,
        schema,
        [(pl.col("end").sum().over("group").alias("group_sum"), "Int64")],
    )
    selected = ["id", "group_sum"]
    expected = [{"id": 1, "group_sum": 43}, {"id": 2, "group_sum": 43}]
    assert_rows(
        table(boto3_session=local_session).collect().select(selected),
        selected,
        expected,
    )
    assert_rows(
        table(columns=selected, boto3_session=local_session).collect(),
        selected,
        expected,
    )


@pytest.mark.parametrize("columns", [["constant"], ["id", "constant"]])
def test_constant_expressions_preserve_row_cardinality(
    tmp_path: Path, local_session: boto3.Session, columns: list[str]
) -> None:
    table = table_at(
        tmp_path, ROWS, SCHEMA, [(pl.lit(7, dtype=pl.Int64).alias("constant"), "Int64")]
    )
    expected = [{"id": 1, "constant": 7}, {"id": 2, "constant": 7}]
    frame = table(columns=columns, boto3_session=local_session).collect()
    assert_rows(frame, columns, expected)
    assert frame.schema["constant"] == pl.Int64


def test_filtered_empty_results_keep_derived_types(
    tmp_path: Path, local_session: boto3.Session
) -> None:
    table = table_at(tmp_path, ROWS, SCHEMA, duration())
    frame = table(
        filters="id = 999", columns=["id", "duration"], boto3_session=local_session
    ).collect()
    assert frame.shape == (0, 2)
    assert frame.schema == {"id": pl.Int64, "duration": pl.Int64}


def test_derived_projection_preserves_complete_compound_identity(
    tmp_path: Path, local_session: boto3.Session
) -> None:
    schema_fields: list[tuple[str, pa.DataType]] = [
        ("subject", pa.string()),
        ("session", pa.int64()),
        ("start", pa.int64()),
        ("end", pa.int64()),
    ]
    schema = pa.schema(schema_fields)
    rows = [
        {"subject": "a", "session": 1, "start": 10, "end": 15},
        {"subject": "a", "session": 1, "start": 10, "end": 15},
        {"subject": "a", "session": 2, "start": 20, "end": 25},
        {"subject": "b", "session": 1, "start": 30, "end": 35},
    ]
    table = table_at(
        tmp_path, rows, schema, duration(), unique_columns=["subject", "session"]
    )
    expected = [{"duration": 5}, {"duration": 5}, {"duration": 5}]
    assert_rows(
        table(boto3_session=local_session).collect().select("duration"),
        ["duration"],
        expected,
    )
    assert_rows(
        table(columns=["duration"], boto3_session=local_session).collect(),
        ["duration"],
        expected,
    )


def test_missing_stored_expression_input_keeps_declared_null_normalization(
    tmp_path: Path, local_session: boto3.Session
) -> None:
    table = table_at(
        tmp_path,
        [{"id": 1, "start": 10}, {"id": 2, "start": 20}],
        SCHEMA,
        duration(),
        stored_schema=pa.schema([("id", pa.int64()), ("start", pa.int64())]),
    )
    expected = [{"id": 1, "duration": None}, {"id": 2, "duration": None}]
    selected = ["id", "duration"]
    assert_rows(
        table(boto3_session=local_session).collect().select(selected),
        selected,
        expected,
    )
    frame = table(columns=selected, boto3_session=local_session).collect()
    assert_rows(frame, selected, expected)
    assert frame.schema["duration"] == pl.Int64


def test_invalid_unrequested_expressions_still_raise(
    tmp_path: Path, local_session: boto3.Session
) -> None:
    table = table_at(
        tmp_path, ROWS, SCHEMA, [(pl.col("missing").alias("bad"), "Int64")]
    )
    for columns in (None, ["id"]):
        with pytest.raises(pl.exceptions.ColumnNotFoundError, match="missing"):
            table(columns=columns, boto3_session=local_session).collect()


def test_schema_sensitive_expression_preserves_missing_requested_column_error(
    tmp_path: Path, local_session: boto3.Session
) -> None:
    table = table_at(
        tmp_path,
        [{"id": 1, "start": 10}, {"id": 2, "start": 20}],
        SCHEMA,
        [(pl.sum_horizontal(pl.all()).alias("total"), "Int64")],
        stored_schema=pa.schema([("id", pa.int64()), ("start", pa.int64())]),
    )
    with pytest.raises(DeltaError, match="No field named end"):
        table(columns=["id", "end"], boto3_session=local_session).collect()


def test_named_expression_projection_does_not_read_unrelated_columns(
    tmp_path: Path, local_session: boto3.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [dict(row, irrelevant="unused") for row in ROWS]
    table = table_at(
        tmp_path, rows, SCHEMA.append(pa.field("irrelevant", pa.string())), duration()
    )
    queries: list[str] = []

    class RecordingBuilder:
        def __init__(self) -> None:
            self.native = QueryBuilder()

        def register(self, name: str, value: DeltaTable) -> "RecordingBuilder":
            self.native = self.native.register(name, value)
            return self

        def execute(self, query: str):
            queries.append(query)
            return self.native.execute(query)

    monkeypatch.setattr(implementation, "QueryBuilder", RecordingBuilder)
    frame = table(columns=["id", "duration"], boto3_session=local_session).collect()
    assert_rows(
        frame, ["id", "duration"], [{"id": 1, "duration": 5}, {"id": 2, "duration": 8}]
    )
    assert len(queries) == 1
    selected = queries[0].split("FROM")[0]
    assert "*" not in selected
    assert '"irrelevant"' not in selected


def test_expression_input_identifiers_are_quoted(
    tmp_path: Path, local_session: boto3.Session
) -> None:
    schema = pa.schema(
        [("id", pa.int64()), ('start"value', pa.int64()), ("end", pa.int64())]
    )
    rows = [
        {"id": 1, 'start"value': 10, "end": 15},
        {"id": 2, 'start"value': 20, "end": 28},
    ]
    table = table_at(
        tmp_path,
        rows,
        schema,
        [((pl.col("end") - pl.col('start"value')).alias("duration"), "Int64")],
    )
    expected = [{"id": 1, "duration": 5}, {"id": 2, "duration": 8}]
    assert_rows(
        table(boto3_session=local_session).collect().select(["id", "duration"]),
        ["id", "duration"],
        expected,
    )
    assert_rows(
        table(columns=["id", "duration"], boto3_session=local_session).collect(),
        ["id", "duration"],
        expected,
    )
