from pathlib import Path

import polars as pl
import pytest
import pyarrow as pa

from datarepo.core.tables.clickhouse_table import ClickHouseTable, ClickHouseTableConfig
from datarepo.core.tables.decorator import table
from datarepo.core.tables.deltalake_table import DeltalakeTable
from datarepo.core.tables.parquet_table import ParquetTable
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
        assert [p["operator"] for p in exported["partitions"]] == ["=", "=", "="]

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
        assert [p["operator"] for p in exported["partitions"]] == ["=", "="]

    def test_export_table_preserves_docs_filter_operator(self):
        table = DeltalakeTable(
            name="test_delta",
            uri="s3://fake-bucket/test/table",
            schema=pa.schema(
                [
                    ("p_name", pa.string()),
                    ("p_size", pa.int32()),
                ]
            ),
            docs_filters=[
                Filter("p_name", "contains", "Brand"),
                Filter("p_size", ">=", 10),
            ],
        )

        exported = export_table("test_delta", table)

        assert [
            (p["column_name"], p["operator"], p["value"])
            for p in exported["partitions"]
        ] == [
            ("p_name", "contains", "Brand"),
            ("p_size", ">=", 10),
        ]

    def test_export_clickhouse_preserves_docs_filter_operator(self, clickhouse_config):
        clickhouse_table = ClickHouseTable(
            name="test_table",
            schema=pa.schema(
                [
                    ("str_value", pa.string()),
                    ("value", pa.int64()),
                ]
            ),
            config=clickhouse_config,
            docs_filters=[Filter("str_value", "contains", "abc")],
        )

        exported = export_table("test_table", clickhouse_table)

        assert exported["partitions"] == [
            {
                "column_name": "str_value",
                "operator": "contains",
                "type_annotation": "string",
                "value": "abc",
            }
        ]

    def test_export_parquet_preserves_docs_filter_operator(self, tmp_path: Path):
        path = tmp_path / "example.parquet"
        pl.DataFrame({"id": [1, 2], "name": ["acme", "other"]}).write_parquet(path)
        parquet_table = ParquetTable(
            name="example",
            uri=str(path),
            partitioning=[],
            docs_filters=[Filter("name", "contains", "acme")],
        )

        exported = export_table("example", parquet_table)

        assert exported["partitions"][0]["column_name"] == "name"
        assert exported["partitions"][0]["operator"] == "contains"
        assert exported["partitions"][0]["value"] == "acme"

    def test_export_function_table_preserves_docs_filter_operator(self):
        @table(docs_args={"filters": [Filter("name", "contains", "acme")]})
        def suppliers():
            return pl.LazyFrame({"name": ["acme", "other"]})

        exported = export_table("suppliers", suppliers)

        assert exported["partitions"][0]["column_name"] == "name"
        assert exported["partitions"][0]["operator"] == "contains"
        assert exported["partitions"][0]["value"] == "acme"

    def test_delta_get_schema_preserves_docs_filter_operator(self):
        delta_table = DeltalakeTable(
            name="test_delta",
            uri="s3://fake-bucket/test/table",
            schema=pa.schema(
                [
                    ("implant_id", pa.int64()),
                    ("date", pa.string()),
                ]
            ),
            partition_columns=["implant_id", "date"],
            docs_filters=[Filter("implant_id", ">=", 1)],
        )

        schema = delta_table.get_schema()

        assert [
            (p["column_name"], p["operator"], p["value"]) for p in schema.partitions
        ] == [
            ("implant_id", ">=", 1),
            ("date", "=", None),
        ]

    def test_export_partition_without_docs_filter_defaults_to_equality(self):
        delta_table = DeltalakeTable(
            name="test_delta",
            uri="s3://fake-bucket/test/table",
            schema=pa.schema(
                [
                    ("implant_id", pa.int64()),
                    ("date", pa.string()),
                ]
            ),
            partition_columns=["implant_id", "date"],
        )

        exported = export_table("test_delta", delta_table)

        assert [
            (p["column_name"], p["operator"], p["value"])
            for p in exported["partitions"]
        ] == [
            ("implant_id", "=", None),
            ("date", "=", None),
        ]
