import unittest

import pyarrow as pa

from datarepo.core import (
    DeltalakeTable,
    Filter,
    ModuleDatabase,
    NlkDataFrame,
    ParquetTable,
    Partition,
    PartitioningScheme,
    table,
)
from datarepo.core.catalog import Catalog
from datarepo.export.roapi import export_to_roapi_tables
from test.roapi.data import database


@table
def new_table() -> NlkDataFrame:
    return NlkDataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})


parquet = ParquetTable(
    name="test_parquet_table",
    uri="s3://bucket/data/",
    partitioning=[Partition(column="implant_id", col_type=pa.int16())],
    docs_filters=[Filter("implant_id", "=", 4595)],
)

parquet_hive = ParquetTable(
    name="test_parquet_table",
    uri="s3://bucket/data/",
    partitioning=[Partition(column="implant_id", col_type=pa.int16())],
    partitioning_scheme=PartitioningScheme.HIVE,
    docs_filters=[Filter("implant_id", "=", 4595)],
)

delta_table = DeltalakeTable(
    name="test_delta_table",
    schema=pa.schema([]),
    uri="s3://bucket/data/",
)


class TestRoapi(unittest.TestCase):
    def setUp(self):
        dbs = {"database": ModuleDatabase(database)}

        self.catalog = Catalog(dbs)

    def test_export_roapi(self):
        result = export_to_roapi_tables(self.catalog)
        result.sort(key=lambda x: x["name"])

        assert result == [
            {
                "name": "database_test_delta_table",
                "uri": "s3://bucket/data/",
                "option": {"format": "delta", "use_memory_table": False},
                "reload_interval": {"secs": 60, "nanos": 0},
            },
            {
                "name": "database_test_parquet_table_hive",
                "uri": "s3://bucket/data/",
                "option": {"format": "parquet", "use_memory_table": False},
                "partition_columns": [
                    {"name": "implant_id", "data_type": "Int64"},
                ],
                "schema_from_files": ["implant_id=4595/df.parquet"],
            },
            {
                "name": "new_name",
                "uri": "s3://bucket/data/",
                "option": {"format": "delta", "use_memory_table": False},
            },
        ]
