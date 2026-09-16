"""A fixed transport stream separates row identity from public projection.

This double supports SELECT projection only. Winner expectations refer to this
stream's last observation; ClickHouse queries do not promise chronological order.
"""

from copy import deepcopy
import re

import polars as pl
import pyarrow as pa
import pytest

from datarepo.core.tables.clickhouse_table import ClickHouseTable, ClickHouseTableConfig


SCHEMA_TYPES: dict[str, pa.DataType] = {
    "subject": pa.string(),
    "session": pa.int64(),
    "value": pa.int64(),
}
SCHEMA = pa.schema(SCHEMA_TYPES)
ROWS = [
    {"subject": "A", "session": 1, "value": 10},
    {"subject": "A", "session": 1, "value": 11},
    {"subject": "A", "session": 2, "value": 20},
    {"subject": "B", "session": 1, "value": 30},
]


class ProjectionStream:
    def __init__(self, rows, source="`default`.`observations`"):
        self.rows = pa.Table.from_pylist(rows, schema=SCHEMA)
        self.source = source
        self.projections = []

    def query_arrow(self, query):
        match = re.fullmatch(
            r"SELECT (.+) FROM " + re.escape(self.source) + r"\s*", query
        )
        assert match, f"Projection-only transport cannot execute {query!r}"
        projection = match.group(1)
        if projection == "*":
            self.projections.append(None)
            return self.rows
        terms = projection.split(", ")
        assert all(re.fullmatch(r"`[a-z_]+`", term) for term in terms), projection
        columns = [term[1:-1] for term in terms]
        self.projections.append(columns)
        return self.rows.select(columns)


@pytest.fixture
def make_table(monkeypatch):
    def make(rows=ROWS, unique_columns=("subject", "session")):
        stream = ProjectionStream(rows)
        monkeypatch.setattr(ClickHouseTableConfig, "get_client", lambda self: stream)
        table = ClickHouseTable(
            name="observations", schema=SCHEMA, unique_columns=list(unique_columns)
        )
        return table, stream

    return make


@pytest.mark.parametrize(
    "columns",
    [
        None,
        [],
        ["subject", "session", "value"],
        ["subject", "value"],
        ["session", "value"],
        ["value"],
        ["value", "subject"],
        ["session"],
        ["value", "session", "subject"],
        ["invalid"],
        ["invalid", "value"],
    ],
)
def test_projection_preserves_full_identity_and_last_observed_rows(make_table, columns):
    table, _ = make_table()
    before = deepcopy(columns)
    public = [c for c in columns or [] if c in SCHEMA.names] or SCHEMA.names
    result = table(columns=columns).collect()
    assert result.columns == public
    assert result.to_dicts() == [{c: row[c] for c in public} for row in ROWS[1:]]
    assert columns == before


@pytest.mark.parametrize(
    "columns,transport",
    [
        (["subject", "value"], ["subject", "value", "session"]),
        (["session", "value"], ["session", "value", "subject"]),
        (["value"], ["value", "subject", "session"]),
        (["value", "subject", "session"], ["value", "subject", "session"]),
        (["invalid", "value"], ["value", "subject", "session"]),
        (["invalid"], None),
        (None, None),
        ([], None),
    ],
)
def test_only_missing_keys_are_added_to_transport(make_table, columns, transport):
    table, stream = make_table()
    table(columns=columns).collect()
    assert stream.projections == [transport]


def test_distinct_full_keys_with_equal_values_and_null_keys_keep_multiplicity(
    make_table,
):
    rows = [
        {"subject": None, "session": 1, "value": 10},
        {"subject": None, "session": 1, "value": 11},
        {"subject": None, "session": 2, "value": 11},
        {"subject": "A", "session": 1, "value": 11},
    ]
    table, _ = make_table(rows)
    assert table(columns=["value"]).collect().to_dicts() == [{"value": 11}] * 3


@pytest.mark.parametrize(
    "columns", [["value"], ["session", "value"], None, ["invalid"]]
)
def test_empty_stream_retains_only_public_schema(make_table, columns):
    table, _ = make_table([])
    result = table(columns=columns).collect()
    public = [c for c in columns or [] if c in SCHEMA.names] or SCHEMA.names
    assert result.height == 0
    assert result.columns == public
    assert (
        result.schema
        == pl.from_arrow(pa.Table.from_pylist([], schema=SCHEMA)).select(public).schema
    )


def test_no_key_means_no_extra_transport_or_deduplication(make_table):
    table, stream = make_table(unique_columns=[])
    assert table(columns=["value"]).collect()["value"].to_list() == [10, 11, 20, 30]
    assert stream.projections == [["value"]]


def test_per_call_config_and_physical_table_override_are_preserved(
    make_table, monkeypatch
):
    table, original = make_table()
    override = ClickHouseTableConfig(
        host="127.0.0.1", database="alternate", table_name="physical"
    )
    alternate = ProjectionStream(ROWS, source="`alternate`.`physical`")
    monkeypatch.setattr(
        ClickHouseTableConfig,
        "get_client",
        lambda self: alternate if self is override else original,
    )
    assert table(columns=["value"], config=override).collect()["value"].to_list() == [
        11,
        20,
        30,
    ]
    assert original.projections == []
    assert alternate.projections == [["value", "subject", "session"]]


def test_incomplete_returned_key_does_not_silently_reduce_identity(
    make_table, monkeypatch
):
    table, stream = make_table()
    monkeypatch.setattr(
        stream, "query_arrow", lambda query: stream.rows.drop(["session"])
    )
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="session"):
        table(columns=["value"]).collect()


def test_invalid_declared_key_is_rejected_on_read_without_eager_connection(monkeypatch):
    def no_connection(self):
        pytest.fail("Invalid key must be rejected before connecting")

    monkeypatch.setattr(ClickHouseTableConfig, "get_client", no_connection)
    table = ClickHouseTable(
        "observations", SCHEMA, unique_columns=["subject", "absent"]
    )
    with pytest.raises(ValueError, match="absent"):
        table(columns=["value"])
