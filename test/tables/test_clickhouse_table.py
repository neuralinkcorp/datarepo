import os

import pytest
from unittest.mock import patch, MagicMock
import polars as pl
import pyarrow as pa

from datarepo.core.tables.clickhouse_table import (
    ClickHouseTable,
    ClickHouseTableConfig,
    make_clickhouse_config,
)
from datarepo.core.tables.filters import Filter
from datarepo.core.tables.metadata import TableSchema


class TestMakeClickHouseConfig:
    def test_defaults_match_http_port(self, monkeypatch: pytest.MonkeyPatch):
        """Unset env uses HTTP 8123, not TLS 8443."""
        for key in (
            "CLICKHOUSE_HOST",
            "CLICKHOUSE_PORT",
            "CLICKHOUSE_USER",
            "CLICKHOUSE_PASSWORD",
            "CLICKHOUSE_DATABASE",
        ):
            monkeypatch.delenv(key, raising=False)

        config = make_clickhouse_config()

        assert config.host == "localhost"
        assert config.port == 8123
        assert config.username == "default"
        assert config.password == ""
        assert config.database == "default"
        assert config.secure is False
        assert config.verify is False


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
            table_name="test_table",
            secure=True,
            verify=True,
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
        assert schema.columns[0]["name"] == "implant_id"
        assert schema.columns[0]["type"] == "int64"
        assert schema.columns[0]["has_stats"] is False
        assert schema.partitions == []

    def test_build_query_no_filters(self, clickhouse_table: ClickHouseTable):
        """Test query building without filters."""
        query = clickhouse_table._build_query()

        expected_query = "SELECT * FROM `test_db`.`test_table` "
        assert query == expected_query

    def test_build_query_falls_back_to_self_name(self):
        """SQL uses ClickHouseTable.name when config.table_name is unset."""
        table = ClickHouseTable(
            name="logical_name",
            schema=pa.schema([("value", pa.int64())]),
            config=ClickHouseTableConfig(
                host="localhost",
                database="test_db",
            ),
        )

        query = table._build_query()

        assert query == "SELECT * FROM `test_db`.`logical_name` "

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

    @patch.dict(
        os.environ,
        {
            "CLICKHOUSE_HOST": "from-env",
            "CLICKHOUSE_PORT": "9000",
            "CLICKHOUSE_USER": "env_user",
            "CLICKHOUSE_PASSWORD": "env_password",
            "CLICKHOUSE_DATABASE": "env_db",
        },
    )
    @patch("clickhouse_connect.get_client")
    def test_call_with_no_filters(
        self, mock_get_client, clickhouse_table: ClickHouseTable
    ):
        """Test calling the table with no filters uses stored config, not env."""
        mock_df = pl.DataFrame(
            {
                "implant_id": [1, 2, 3],
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "value": [10, 20, 30],
            }
        )
        mock_client = MagicMock()
        mock_client.query_arrow.return_value = mock_df.to_arrow()
        mock_get_client.return_value = mock_client

        result = clickhouse_table().collect()

        mock_get_client.assert_called_once_with(
            host="localhost",
            port=8443,
            username="test_user",
            password="test_password",
            database="test_db",
            secure=True,
            verify=True,
            settings={},
        )
        mock_client.query_arrow.assert_called_once_with(
            "SELECT * FROM `test_db`.`test_table` "
        )

        assert result.equals(mock_df)

    @patch("clickhouse_connect.get_client")
    def test_call_with_filters_and_columns(
        self, mock_get_client: MagicMock, clickhouse_table: ClickHouseTable
    ):
        """Test calling the table with filters and columns."""
        mock_df = pl.DataFrame(
            {
                "implant_id": [1],
                "value": [10],
            }
        )
        mock_client = MagicMock()
        mock_client.query_arrow.return_value = mock_df.to_arrow()
        mock_get_client.return_value = mock_client

        filters = [Filter("implant_id", "=", 1)]
        columns = ["implant_id", "value"]
        result = clickhouse_table(filters=filters, columns=columns).collect()

        mock_get_client.assert_called_once_with(
            host="localhost",
            port=8443,
            username="test_user",
            password="test_password",
            database="test_db",
            secure=True,
            verify=True,
            settings={},
        )
        mock_client.query_arrow.assert_called_once_with(
            "SELECT `implant_id`, `value` FROM `test_db`.`test_table` WHERE (`implant_id` = 1)"
        )

        assert result.equals(mock_df)

    @patch("clickhouse_connect.get_client")
    def test_call_handles_empty_results(
        self, mock_get_client: MagicMock, clickhouse_table: ClickHouseTable
    ):
        """Test that the table handles empty results correctly."""
        mock_df = pl.DataFrame(
            schema={
                "implant_id": pl.Int64,
                "date": pl.Utf8,
                "value": pl.Int64,
            }
        )
        mock_client = MagicMock()
        mock_client.query_arrow.return_value = mock_df.to_arrow()
        mock_get_client.return_value = mock_client

        filters = [Filter("implant_id", "=", 999)]
        result = clickhouse_table(filters=filters).collect()

        assert result.height == 0
        assert "implant_id" in result.columns
        assert "date" in result.columns
        assert "value" in result.columns

    @patch("clickhouse_connect.get_client")
    def test_call_uses_override_config(
        self, mock_get_client: MagicMock, clickhouse_table: ClickHouseTable
    ):
        """An explicit config argument wins over the table's stored config."""
        mock_df = pl.DataFrame({"implant_id": [1]})
        mock_client = MagicMock()
        mock_client.query_arrow.return_value = mock_df.to_arrow()
        mock_get_client.return_value = mock_client

        override = ClickHouseTableConfig(
            host="other-host",
            port=8123,
            username="other_user",
            password="other_password",
            database="other_db",
            table_name="other_table",
        )
        clickhouse_table(config=override).collect()

        mock_get_client.assert_called_once_with(
            host="other-host",
            port=8123,
            username="other_user",
            password="other_password",
            database="other_db",
            secure=False,
            verify=False,
            settings={},
        )
        mock_client.query_arrow.assert_called_once_with(
            "SELECT * FROM `other_db`.`other_table` "
        )
