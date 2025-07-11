import pytest
from unittest.mock import patch, MagicMock
import polars as pl
import pyarrow as pa

from datarepo.core.tables.clickhouse_table import ClickHouseTable, ClickHouseTableConfig
from datarepo.core.tables.filters import Filter
from datarepo.core.tables.metadata import TableSchema


class TestClickHouseTable:
    @pytest.fixture
    def clickhouse_config(self):
        """Create a test ClickHouseTableConfig."""
        return ClickHouseTableConfig(
            host="localhost",
            port=8443,
            username="test_user",
            password="test_password",
            database="test_db",
        )

    @pytest.fixture
    def clickhouse_table(self, clickhouse_config):
        """Create a test ClickHouseTable."""
        return ClickHouseTable(
            name="test_table",
            schema=pa.schema(
                [
                    ("implant_id", pa.int64()),
                    ("date", pa.string()),
                    ("value", pa.int64()),
                    ("str_value", pa.string()),
                    ("arr_value", pa.list_(pa.int64())),
                ]
            ),
            config=clickhouse_config,
            description="Test ClickHouse table",
            unique_columns=["implant_id", "date"],
        )

    def test_get_schema(self, clickhouse_table: ClickHouseTable):
        """Test that get_schema returns the correct schema."""
        schema = clickhouse_table.get_schema()

        assert isinstance(schema, TableSchema)
        assert len(schema.columns) == 5
        assert schema.columns[0]["column"] == "implant_id"
        assert schema.columns[0]["type"] == "int64"
        assert schema.columns[0]["has_stats"] is False
        assert schema.partitions == []

    def test_build_query_no_filters(self, clickhouse_table: ClickHouseTable):
        """Test query building without filters."""
        query = clickhouse_table._build_query()

        expected_query = "SELECT * FROM `test_db`.`test_table` "
        assert query == expected_query

    def test_build_query_with_columns(self, clickhouse_table: ClickHouseTable):
        """Test query building with specific columns."""
        query = clickhouse_table._build_query(columns=["implant_id", "value"])

        expected_query = "SELECT `implant_id`, `value` FROM `test_db`.`test_table` "
        assert query == expected_query

    def test_build_query_with_filters(self, clickhouse_table: ClickHouseTable):
        """Test query building with filters."""
        filters = [Filter("implant_id", "=", 1)]
        query = clickhouse_table._build_query(filters=filters)

        expected_query = "SELECT * FROM `test_db`.`test_table` WHERE (`implant_id` = 1)"
        assert query == expected_query

    def test_build_query_with_multiple_filters(self, clickhouse_table: ClickHouseTable):
        """Test query building with multiple filters."""
        filters = [
            [Filter("implant_id", "=", 1), Filter("date", "=", "2023-01-01")],
            [Filter("value", ">", 50)],
        ]
        query = clickhouse_table._build_query(filters=filters)

        expected_query = "SELECT * FROM `test_db`.`test_table` WHERE (`implant_id` = 1 AND `date` = '2023-01-01') OR (`value` > 50)"
        assert query == expected_query

    @pytest.mark.parametrize(
        "operator,value,expected_condition",
        [
            ("=", 1, "`implant_id` = 1"),
            ("!=", 1, "`implant_id` != 1"),
            (">", 1, "`implant_id` > 1"),
            ("<", 1, "`implant_id` < 1"),
            (">=", 1, "`implant_id` >= 1"),
            ("<=", 1, "`implant_id` <= 1"),
            ("in", [1, 2, 3], "`implant_id` IN (1, 2, 3)"),
            ("not in", [1, 2, 3], "`implant_id` NOT IN (1, 2, 3)"),
            ("contains", "%test%", "`str_value` LIKE '%test%'"),
        ],
    )
    def test_filter_operators(
        self,
        clickhouse_table: ClickHouseTable,
        operator: str,
        value: str,
        expected_condition: str,
    ):
        """Test different filter operators."""
        column = "str_value" if operator == "contains" else "implant_id"
        filters = [Filter(column, operator, value)]

        query = clickhouse_table._build_query(filters=filters)
        expected_query = (
            f"SELECT * FROM `test_db`.`test_table` WHERE ({expected_condition})"
        )
        assert query == expected_query

    @patch("polars.read_database_uri")
    def test_call_with_no_filters(
        self, mock_read_database_uri, clickhouse_table: ClickHouseTable
    ):
        """Test calling the table with no filters."""
        mock_df = pl.DataFrame(
            {
                "implant_id": [1, 2, 3],
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "value": [10, 20, 30],
            }
        )
        mock_read_database_uri.return_value = mock_df

        result = clickhouse_table().collect()

        mock_read_database_uri.assert_called_once()
        call_args = mock_read_database_uri.call_args[1]
        assert call_args["query"] == "SELECT * FROM `test_db`.`test_table` "
        assert (
            call_args["uri"]
            == "clickhouse://test_user:test_password@localhost:8443/test_db"
        )
        assert call_args["engine"] == "connectorx"

        assert result.equals(mock_df)

    @patch("polars.read_database_uri")
    def test_call_with_filters_and_columns(
        self, mock_read_database_uri: str, clickhouse_table: ClickHouseTable
    ):
        """Test calling the table with filters and columns."""
        mock_df = pl.DataFrame(
            {
                "implant_id": [1],
                "value": [10],
            }
        )
        mock_read_database_uri.return_value = mock_df

        filters = [Filter("implant_id", "=", 1)]
        columns = ["implant_id", "value"]
        result = clickhouse_table(filters=filters, columns=columns).collect()

        mock_read_database_uri.assert_called_once()
        call_args = mock_read_database_uri.call_args[1]
        assert (
            call_args["query"]
            == "SELECT `implant_id`, `value` FROM `test_db`.`test_table` WHERE (`implant_id` = 1)"
        )
        assert (
            call_args["uri"]
            == "clickhouse://test_user:test_password@localhost:8443/test_db"
        )
        assert call_args["engine"] == "connectorx"

        assert result.equals(mock_df)

    @patch("polars.read_database_uri")
    def test_call_handles_empty_results(
        self, mock_read_database_uri: str, clickhouse_table: ClickHouseTable
    ):
        """Test that the table handles empty results correctly."""
        mock_df = pl.DataFrame(
            schema={
                "implant_id": pl.Int64,
                "date": pl.Utf8,
                "value": pl.Int64,
            }
        )
        mock_read_database_uri.return_value = mock_df

        filters = [Filter("implant_id", "=", 999)]
        result = clickhouse_table(filters=filters).collect()

        assert result.height == 0
        assert "implant_id" in result.columns
        assert "date" in result.columns
        assert "value" in result.columns
