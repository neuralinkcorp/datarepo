"""Regression tests for ClickHouse filter predicates and existing grouping rules."""

from typing import cast
from unittest.mock import patch

import pyarrow as pa
import pytest

from datarepo.core.tables.clickhouse_table import ClickHouseTable, ClickHouseTableConfig
from datarepo.core.tables.filters import Filter, FilterOperator


@pytest.fixture
def table():
    return ClickHouseTable(
        name="filter_rows",
        schema=pa.schema(
            [("id", pa.int64()), ("x", pa.int64()), ("label", pa.string())]
        ),
        config=ClickHouseTableConfig(host="127.0.0.1", database="default"),
    )


def where_clause(query):
    """Ignore formatting outside the predicate; keep grouping observable."""
    return query.partition("WHERE")[2].strip()


@pytest.mark.parametrize("operator", ["is null", "is not null"])
@pytest.mark.parametrize("placeholder", [None, "unused'placeholder", [], [1, 2]])
def test_null_predicates_ignore_placeholder(table, operator, placeholder):
    query = table._build_query([Filter("x", operator, placeholder)])
    assert where_clause(query) == f"(`x` {operator.upper()})"


@pytest.mark.parametrize("operator", ["is null", "is not null"])
def test_null_predicates_do_not_format_value(table, operator):
    class Unformattable:
        def __str__(self):
            raise AssertionError("Unary predicates must not evaluate Filter.value")

    with patch(
        "datarepo.core.tables.clickhouse_table.format_value_for_sql",
        side_effect=AssertionError("Unary predicates must not format Filter.value"),
    ) as formatter:
        query = table._build_query([Filter("x", operator, Unformattable())])
    formatter.assert_not_called()
    assert where_clause(query) == f"(`x` {operator.upper()})"


@pytest.mark.parametrize(
    "filters,expected",
    [
        (
            [Filter("x", "is null", None), Filter("id", "=", 1)],
            "(`x` IS NULL AND `id` = 1)",
        ),
        (
            [[Filter("x", "is null", None)], [Filter("id", "=", 2)]],
            "(`x` IS NULL) OR (`id` = 2)",
        ),
        (
            [
                [Filter("x", "is not null", None), Filter("id", ">", 2)],
                [Filter("x", "is null", None), Filter("id", "=", 1)],
            ],
            "(`x` IS NOT NULL AND `id` > 2) OR (`x` IS NULL AND `id` = 1)",
        ),
    ],
)
def test_null_predicates_preserve_and_or_groups(table, filters, expected):
    assert where_clause(table._build_query(filters)) == expected


UNKNOWN_FILTER = Filter("x", cast(FilterOperator, "unsupported"), None)


@pytest.mark.parametrize(
    "filters",
    [
        [UNKNOWN_FILTER],
        [Filter("id", "=", 1), UNKNOWN_FILTER],
        [[Filter("id", "=", 1)], [UNKNOWN_FILTER]],
        [[], [Filter("id", "=", 1), UNKNOWN_FILTER]],
    ],
)
def test_unknown_operator_raises_before_client_creation(table, filters):
    with patch("clickhouse_connect.get_client") as get_client:
        get_client.return_value.query_arrow.return_value = pa.table({"id": [1]})
        with pytest.raises(
            ValueError, match="Unsupported filter operator.*unsupported"
        ):
            table(filters=filters)
    get_client.assert_not_called()


@pytest.mark.parametrize("filters", [None, [], [[]], [[], []]])
def test_existing_empty_filter_behavior(table, filters):
    assert where_clause(table._build_query(filters)) == ""


def test_empty_group_alongside_nonempty_group_is_ignored(table):
    filters = [[], [Filter("id", "=", 2)], []]
    assert where_clause(table._build_query(filters)) == "(`id` = 2)"


@pytest.mark.parametrize(
    "operator,value,expected",
    [
        ("=", None, "`x` = NULL"),
        ("!=", 2, "`x` != 2"),
        ("<", 2, "`x` < 2"),
        ("<=", 2, "`x` <= 2"),
        (">", 2, "`x` > 2"),
        (">=", 2, "`x` >= 2"),
        ("in", [1, 2], "`x` IN (1, 2)"),
        ("not in", [1, 2], "`x` NOT IN (1, 2)"),
        ("in", [], "`x` IN ()"),
        ("not in", [], "`x` NOT IN ()"),
        ("contains", "%cat%", "`x` LIKE '%cat%'"),
        ("includes", "%cat%", "`x` LIKE '%cat%'"),
        ("includes any", "%cat%", "`x` LIKE '%cat%'"),
        ("includes all", "%cat%", "`x` LIKE '%cat%'"),
    ],
)
def test_recognized_operator_behavior_is_unchanged(table, operator, value, expected):
    assert where_clause(table._build_query([Filter("x", operator, value)])) == (
        f"({expected})"
    )


def test_existing_invalid_column_behavior_is_unchanged(table):
    query = table._build_query([Filter("unknown", "=", 2)], columns=["unknown"])
    assert query.startswith("SELECT * FROM")
    assert where_clause(query) == "(`unknown` = 2)"
