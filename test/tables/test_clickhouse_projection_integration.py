"""Opt-in full-key cardinality checks against a disposable loopback table.

Set DATAREPO_TEST_CLICKHOUSE_PORT to require this backend. Missing/unavailable
requested backends fail. Duplicate versions have equal values, so these checks
make no assumption about ClickHouse's unordered duplicate winner.
"""

from collections import Counter
import os
from uuid import uuid4

import pyarrow as pa
import pytest

from datarepo.core.tables.clickhouse_table import ClickHouseTable, ClickHouseTableConfig
from datarepo.core.tables.filters import Filter


EXPECTED = [("A", 1, 11), ("A", 2, 11), ("B", 1, 30), (None, 1, 40), (None, 2, 40)]
NAMES = ["subject", "session", "value"]
SCHEMA_TYPES: dict[str, pa.DataType] = {
    "subject": pa.string(),
    "session": pa.int64(),
    "value": pa.int64(),
}
SCHEMA = pa.schema(SCHEMA_TYPES)


@pytest.fixture(scope="module")
def backend_table():
    port = os.environ.get("DATAREPO_TEST_CLICKHOUSE_PORT")
    if port is None:
        pytest.skip(
            "set DATAREPO_TEST_CLICKHOUSE_PORT for local ClickHouse integration"
        )
    name = f"datarepo_full_key_{uuid4().hex}"
    config = ClickHouseTableConfig(host="127.0.0.1", port=int(port), table_name=name)
    client = config.get_client()
    created = False
    try:
        client.command(
            f"CREATE TABLE `{name}` (subject Nullable(String), session Int64, value Int64) ENGINE = Memory"
        )
        created = True
        client.insert(name, EXPECTED + [EXPECTED[0]], column_names=NAMES)
        print(f"ClickHouse version: {client.command('SELECT version()')}")
        yield ClickHouseTable(
            "logical_observations",
            SCHEMA,
            config=config,
            unique_columns=["subject", "session"],
        )
    finally:
        if created:
            client.command(f"DROP TABLE `{name}`")
        client.close()


@pytest.mark.parametrize(
    "columns",
    [
        None,
        NAMES,
        ["subject", "value"],
        ["session", "value"],
        ["value"],
        ["value", "subject"],
        ["invalid"],
        ["invalid", "value"],
    ],
)
def test_local_projection_preserves_full_key_cardinality(backend_table, columns):
    public = [c for c in columns or [] if c in NAMES] or NAMES
    expected = Counter(tuple(row[NAMES.index(c)] for c in public) for row in EXPECTED)
    result = backend_table(columns=columns).collect()
    assert result.columns == public
    assert Counter(result.rows()) == expected


def test_local_null_filter_keeps_distinct_null_keys(backend_table):
    result = backend_table(
        filters=[Filter("subject", "is null", None)], columns=["value"]
    ).collect()
    assert result.columns == ["value"]
    assert result["value"].to_list() == [40, 40]


def test_local_empty_result_hides_transport_keys(backend_table):
    result = backend_table(
        filters=[Filter("value", ">", 100)], columns=["value"]
    ).collect()
    assert result.height == 0
    assert result.columns == ["value"]


def test_local_no_key_does_not_deduplicate(backend_table):
    table = ClickHouseTable("logical_observations", SCHEMA, config=backend_table.config)
    assert table(columns=["value"]).collect().height == 6
