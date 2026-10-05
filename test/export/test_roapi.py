from datetime import date

import pytest
import pyarrow as pa

from datarepo.core.tables.clickhouse_table import ClickHouseTable, ClickHouseTableConfig
from datarepo.core.tables.parquet_table import ParquetTable
from datarepo.core.tables.util import Filter, Partition, PartitioningScheme
from datarepo.export.roapi import export_to_roapi_table, py_type_to_roapi


class TestRoapiExport:
    @pytest.fixture
    def parquet_table(self):
        """Create a test ParquetTable with Hive partitioning."""
        return ParquetTable(
            name="test_parquet_table",
            uri="s3://test-bucket/data/",
            partitioning=[
                Partition(column="implant_id", col_type=pa.int64()),
                Partition(column="date", col_type=pa.string()),
            ],
            partitioning_scheme=PartitioningScheme.HIVE,
            docs_filters=[
                Filter("implant_id", "=", 12345),
                Filter("date", "=", "2024-10-29"),
            ],
            description="Test Parquet table for ROAPI export",
        )

    @pytest.fixture
    def clickhouse_table(self):
        """Create a test ClickHouseTable."""
        config = ClickHouseTableConfig(
            host="localhost",
            port=8443,
            username="test_user",
            password="test_password",
            database="test_db",
        )
        return ClickHouseTable(
            name="test_clickhouse_table",
            schema=pa.schema(
                [
                    ("implant_id", pa.int64()),
                    ("date", pa.string()),
                    ("value", pa.int64()),
                ]
            ),
            config=config,
            description="Test ClickHouse table for ROAPI export",
        )

    def test_roapi_export_parquet_partition_columns(self, parquet_table: ParquetTable):
        """Test that ROAPI export uses 'name' field for partition columns."""
        exported = export_to_roapi_table("test_table", parquet_table)

        assert exported is not None
        assert "partition_columns" in exported
        assert len(exported["partition_columns"]) == 2

        # Verify that partition columns use 'name' field
        for partition_col in exported["partition_columns"]:
            assert "name" in partition_col, "Partition column should have 'name' field"
            assert (
                "column" not in partition_col
            ), "Partition column should NOT have 'column' field"

        # Verify the actual partition column names
        partition_names = [col["name"] for col in exported["partition_columns"]]
        assert partition_names == ["implant_id", "date"]

        # Verify data types are present
        assert exported["partition_columns"][0]["data_type"] == "Int64"
        assert exported["partition_columns"][1]["data_type"] == "Date32"  # date is special-cased

    def test_roapi_export_clickhouse_table_returns_none(self, clickhouse_table: ClickHouseTable):
        """Test that ROAPI export skips ClickHouse tables (unsupported format)."""
        exported = export_to_roapi_table("test_clickhouse_table", clickhouse_table)

        assert exported is None

    def test_roapi_export_unsupported_partition_type_raises_clear_error(self):
        """A partition value type with no Roapi mapping should raise a clear
        ValueError naming the type, instead of an opaque KeyError."""
        table = ParquetTable(
            name="test_parquet_table",
            uri="s3://test-bucket/data/",
            partitioning=[Partition(column="created_on", col_type=pa.date32())],
            partitioning_scheme=PartitioningScheme.HIVE,
            docs_filters=[Filter("created_on", "=", date(2024, 10, 29))],
        )

        with pytest.raises(ValueError, match="Unsupported partition column type"):
            export_to_roapi_table("test_table", table)

    @pytest.mark.parametrize(
        ("py_type", "expected"),
        [(int, "Int64"), (str, "Utf8"), (bool, "Boolean"), (float, "Float64")],
    )
    def test_py_type_to_roapi_supported_types(self, py_type: type, expected: str):
        assert py_type_to_roapi(py_type) == expected

    def test_py_type_to_roapi_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported partition column type"):
            py_type_to_roapi(date)
