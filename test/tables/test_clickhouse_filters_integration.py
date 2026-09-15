"""Opt-in local ClickHouse checks; set DATAREPO_TEST_CLICKHOUSE_PORT to run.

Only a uniquely named disposable table on 127.0.0.1 is created and dropped.
An explicitly requested but unavailable backend fails instead of skipping.
"""

import os
from uuid import uuid4

import pyarrow as pa
import pytest

from datarepo.core.tables.clickhouse_table import ClickHouseTable, ClickHouseTableConfig
from datarepo.core.tables.filters import Filter


@pytest.fixture(scope="module")
def local_table():
    port = os.environ.get("DATAREPO_TEST_CLICKHOUSE_PORT")
    if port is None:
        pytest.skip(
            "set DATAREPO_TEST_CLICKHOUSE_PORT for local ClickHouse integration"
        )
    name = f"datarepo_null_filters_{uuid4().hex}"
    config = ClickHouseTableConfig(
        host="127.0.0.1", port=int(port), database="default", table_name=name
    )
    client = config.get_client()
    created = False
    try:
        client.command(
            f"CREATE TABLE `{name}` (id Int64, x Nullable(Int64)) ENGINE = Memory"
        )
        created = True
        client.insert(
            name, [(1, None), (2, 10), (3, None), (4, 20)], column_names=["id", "x"]
        )
        print(f"ClickHouse version: {client.command('SELECT version()')}")
        yield ClickHouseTable(
            name=name,
            schema=pa.schema([("id", pa.int64()), ("x", pa.int64())]),
            config=config,
        )
    finally:
        if created:
            client.command(f"DROP TABLE `{name}`")
        client.close()


@pytest.mark.parametrize(
    "filters,expected_ids",
    [
        ([Filter("x", "is null", None)], [1, 3]),
        ([Filter("x", "is not null", None)], [2, 4]),
        ([Filter("x", "is null", None), Filter("id", "=", 1)], [1]),
        ([[Filter("x", "is null", None)], [Filter("id", "=", 2)]], [1, 2, 3]),
        (None, [1, 2, 3, 4]),
    ],
    ids=["is-null", "is-not-null", "null-and-id", "null-or-id", "no-filters"],
)
def test_local_null_filter_rows(local_table, filters, expected_ids):
    rows = local_table(filters=filters, columns=["id"]).collect()
    actual = sorted(rows["id"].to_list())
    print(f"{local_table._build_query(filters, ['id'])}: {actual}")
    assert actual == expected_ids
