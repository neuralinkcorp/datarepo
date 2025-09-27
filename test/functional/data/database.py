import polars as pl
import pyarrow as pa

from datarepo.core import (
    DeltalakeTable,
    NlkDataFrame,
    ParquetTable,
    Partition,
    PartitioningScheme,
    table,
)

frame1 = pl.LazyFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
frame2 = pl.LazyFrame({"a": ["a", "b", "c"], "b": ["d", "e", "f"]})
frame4 = pl.LazyFrame({"z": ["1", "2", "3"], "implant_id": [1, 2, 3]})


@table
def new_table() -> NlkDataFrame:
    return frame1


@table
def new_table_2() -> NlkDataFrame:
    return frame2


def not_a_table(a) -> int:
    return a * 2


@table(is_deprecated=True)
def deprecated_table() -> NlkDataFrame:
    return frame1


my_parquet_table = ParquetTable(
    name="my_parquet_table",
    uri="s3://bucket/data/",
    partitioning=[
        Partition(column="implant_id", col_type=pa.int16()),
        Partition(column="date", col_type=pa.string()),
    ],
)

my_parquet_table_hive = ParquetTable(
    name="my_parquet_table_hive",
    uri="s3://bucket/data/",
    partitioning=[
        Partition(column="implant_id", col_type=pa.int16()),
        Partition(column="date", col_type=pa.string()),
    ],
    partitioning_scheme=PartitioningScheme.HIVE,
)

my_delta_table = DeltalakeTable(
    name="my_delta_table",
    schema=pa.schema(
        [
            ("z", pa.string()),
            ("implant_id", pa.int64()),
            ("date", pa.string()),
        ]
    ),
    uri="s3://bucket/data/",
    unique_columns=["z"],
)
