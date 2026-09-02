import pytest
import pyarrow as pa

from datarepo.core.tables.clickhouse_table import ClickHouseTable, ClickHouseTableConfig
from datarepo.core.tables.deltalake_table import DeltalakeTable
from datarepo.core.tables.util import Filter
from datarepo.export.web import export_table


class TestWebExport:
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
                ]
            ),
            config=clickhouse_config,
            description="Test ClickHouse table for web export",
        )

    def test_export_table_column_name_field(self, clickhouse_table: ClickHouseTable):
        """Test that export_table stores column names in 'name' field, not 'column' field."""
        exported = export_table("test_table", clickhouse_table)

        # Assert that columns are present
        assert "columns" in exported
        assert len(exported["columns"]) == 3

        # Verify the actual column names
        column_names = [col["name"] for col in exported["columns"]]
        assert column_names == ["implant_id", "date", "value"]

        # Verify other column properties are present
        assert exported["columns"][0]["type"] == "int64"
        assert exported["columns"][0]["readonly"] is False
        assert exported["columns"][0]["has_stats"] is False

    def test_export_table_derives_schema_from_attribute(self):
        """Test that export_table derives columns from the in-repo schema attribute
        without hitting S3."""
        table = DeltalakeTable(
            name="test_delta",
            uri="s3://fake-bucket/test/table",
            schema=pa.schema(
                [
                    ("implant_id", pa.int64()),
                    ("date", pa.string()),
                    ("hour", pa.int32()),
                    ("value", pa.float64()),
                ]
            ),
            description="Test delta table",
            docs_filters=[
                Filter("implant_id", "=", 12345),
                Filter("date", "=", "2025-07-21"),
                Filter("hour", "=", 18),
            ],
        )

        exported = export_table("test_delta", table)

        # Columns should be derived from the schema attribute, not from S3
        assert len(exported["columns"]) == 4
        column_names = [col["name"] for col in exported["columns"]]
        assert column_names == ["implant_id", "date", "hour", "value"]
        assert exported["columns"][3]["type"] == "double"

        # Partitions should be inferred from docs_filters
        assert len(exported["partitions"]) == 3
        partition_names = [p["column_name"] for p in exported["partitions"]]
        assert partition_names == ["implant_id", "date", "hour"]
        assert exported["partitions"][0]["value"] == 12345

    def test_export_table_delta_without_docs_filters(self):
        """Test that export_table works for delta tables without docs_filters."""
        table = DeltalakeTable(
            name="test_delta",
            uri="s3://fake-bucket/test/table",
            schema=pa.schema(
                [
                    ("id", pa.int64()),
                    ("data", pa.string()),
                ]
            ),
            description="Test delta table",
        )

        exported = export_table("test_delta", table)

        assert len(exported["columns"]) == 2
        assert len(exported["partitions"]) == 0

    def test_export_table_uses_partition_columns_when_set(self):
        """partition_columns wins over docs_filters when inferring partitions."""
        table = DeltalakeTable(
            name="test_delta",
            uri="s3://fake-bucket/test/table",
            schema=pa.schema(
                [
                    ("implant_id", pa.int64()),
                    ("date", pa.string()),
                    ("hour", pa.int32()),
                    ("value", pa.float64()),
                ]
            ),
            description="Test delta table",
            partition_columns=["implant_id", "date"],
            docs_filters=[
                Filter("implant_id", "=", 12345),
                Filter("date", "=", "2025-07-21"),
                Filter("hour", "=", 18),
            ],
        )

        exported = export_table("test_delta", table)

        partition_names = [p["column_name"] for p in exported["partitions"]]
        assert partition_names == ["implant_id", "date"]
        assert exported["partitions"][0]["value"] == 12345
        assert exported["partitions"][1]["value"] == "2025-07-21"
