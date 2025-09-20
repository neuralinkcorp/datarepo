from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from datarepo.core.tables.filters import (
    Filter,
    InputFilters,
    NormalizedFilters,
)
from datarepo.core.tables.parquet_table import ParquetTable
from datarepo.core.tables.util import Partition, PartitioningScheme


@pytest.fixture
def parquet_table(tmp_path: Path):
    table_dir = tmp_path / "test"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "test.parquet"

    source = pl.DataFrame(
        {
            "implant_id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )
    source.write_parquet(table_path)

    return ParquetTable(name="test", uri=str(table_path), partitioning=[])


@pytest.fixture
def partitioned_parquet_table(tmp_path: Path):
    table_path = tmp_path / "partitioned"

    source = pl.DataFrame(
        {
            "implant_id": [1] * 3 + [2] * 3 + [3] * 3,
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"] * 3,
            # NOTE: value should be unique since we use it to sort the results
            "value": [10, 20, 30, 40, 50, 60, 70, 80, 90],
            "str_value": [
                "test1",
                "test2",
                "test3",
                "abc1",
                "abc2",
                "abc3",
                "xyz1",
                "xyz2",
                "xyz3",
            ],
            "arr_value": [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [1, 0, 1],
                [0, 0, 0],
                [1, 1, 1],
            ],
            "date_time": [
                datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 2, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 3, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 8, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 2, 8, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 3, 8, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 16, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 2, 16, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 3, 16, 0, tzinfo=timezone.utc),
            ],
        }
    )
    for (implant_id, date), df in source.group_by("implant_id", "date"):
        dir = table_path / f"implant_id={implant_id}" / f"date={date}"
        dir.mkdir(parents=True, exist_ok=True)

        df.write_parquet(
            dir / "data.parquet",
            compression="snappy",
        )

    return ParquetTable(
        name="partitioned",
        uri=str(table_path),
        partitioning=[Partition("implant_id", pl.Int32), Partition("date", pl.String)],
        partitioning_scheme=PartitioningScheme.HIVE,
    )


class TestParquetTable:
    @pytest.mark.parametrize(
        ("filters", "expected"),
        [
            # Even though table is not partitioned, filters should work on the results
            (None, pl.DataFrame({"implant_id": [1, 2, 3], "value": [10, 20, 30]})),
            (
                [Filter("implant_id", "=", 1)],
                pl.DataFrame({"implant_id": [1], "value": [10]}),
            ),
            (
                [Filter("value", ">=", 20)],
                pl.DataFrame({"implant_id": [2, 3], "value": [20, 30]}),
            ),
        ],
    )
    def test_read(
        self,
        parquet_table: ParquetTable,
        filters: InputFilters,
        expected: pl.DataFrame,
    ):
        result = parquet_table(filters=filters).collect()

        assert (
            result.sort("value").select(expected.columns).equals(expected.sort("value"))
        )

    @pytest.mark.parametrize(
        (
            "filters",
            "expected",
        ),
        [
            # No filters
            (
                None,
                pl.DataFrame(
                    {
                        "implant_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                        "date": ["2023-01-01", "2023-01-02", "2023-01-03"] * 3,
                        "value": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                    }
                ),
            ),
            # Empty filters
            (
                [],
                pl.DataFrame(
                    {
                        "implant_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                        "date": ["2023-01-01", "2023-01-02", "2023-01-03"] * 3,
                        "value": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                    }
                ),
            ),
            # Filter by one partition
            (
                [
                    Filter("implant_id", "=", 1),
                ],
                pl.DataFrame(
                    {
                        "implant_id": [1, 1, 1],
                        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                        "value": [10, 20, 30],
                    }
                ),
            ),
            # Filter by both partitions
            (
                [
                    Filter("implant_id", "=", 1),
                    Filter("date", "=", "2023-01-01"),
                ],
                pl.DataFrame(
                    {
                        "implant_id": [1],
                        "date": ["2023-01-01"],
                        "value": [10],
                    }
                ),
            ),
            # Multiple partition values
            (
                [
                    Filter("implant_id", "in", [1, 2]),
                    Filter("date", "in", ["2023-01-01", "2023-01-02"]),
                ],
                pl.DataFrame(
                    {
                        "implant_id": [1, 1, 2, 2],
                        "date": ["2023-01-01", "2023-01-02"] * 2,
                        "value": [10, 20, 40, 50],
                    }
                ),
            ),
            # Test all operators
            (
                [
                    Filter("value", ">", 50),
                ],
                pl.DataFrame(
                    {
                        "implant_id": [2, 3, 3, 3],
                        "date": [
                            "2023-01-03",
                            "2023-01-01",
                            "2023-01-02",
                            "2023-01-03",
                        ],
                        "value": [60, 70, 80, 90],
                    }
                ),
            ),
            (
                [
                    Filter("str_value", "contains", "test"),
                ],
                pl.DataFrame(
                    {
                        "implant_id": [1, 1, 1],
                        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                        "value": [10, 20, 30],
                        "str_value": ["test1", "test2", "test3"],
                    }
                ),
            ),
            (
                [
                    Filter("arr_value", "includes", 1),
                ],
                pl.DataFrame(
                    {
                        "implant_id": [1, 2, 3, 3],
                        "date": [
                            "2023-01-01",
                            "2023-01-01",
                            "2023-01-01",
                            "2023-01-03",
                        ],
                        "value": [10, 40, 70, 90],
                        "arr_value": [[1, 2, 3], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
                    }
                ),
            ),
            (
                [
                    Filter("arr_value", "includes any", [1, 2]),
                ],
                pl.DataFrame(
                    {
                        "implant_id": [1, 2, 2, 3, 3],
                        "date": [
                            "2023-01-01",
                            "2023-01-01",
                            "2023-01-02",
                            "2023-01-01",
                            "2023-01-03",
                        ],
                        "value": [10, 40, 50, 70, 90],
                        "arr_value": [
                            [1, 2, 3],
                            [1, 1, 1],
                            [2, 2, 2],
                            [1, 0, 1],
                            [1, 1, 1],
                        ],
                    }
                ),
            ),
            (
                [
                    Filter("arr_value", "includes all", [0, 1]),
                ],
                pl.DataFrame(
                    {
                        "implant_id": [3],
                        "date": ["2023-01-01"],
                        "value": [70],
                        "arr_value": [[1, 0, 1]],
                    }
                ),
            ),
            # Multiple lists of filters should be ORed
            (
                [
                    [Filter("implant_id", "=", 1)],
                    [Filter("date", "=", "2023-01-01")],
                ],
                pl.DataFrame(
                    {
                        "implant_id": [1, 1, 1, 2, 3],
                        "date": [
                            "2023-01-01",
                            "2023-01-02",
                            "2023-01-03",
                            "2023-01-01",
                            "2023-01-01",
                        ],
                        "value": [10, 20, 30, 40, 70],
                    }
                ),
            ),
            # Dates can be passed in filters
            (
                [
                    Filter(
                        "date_time",
                        ">",
                        datetime(2023, 1, 1, 8, 0, tzinfo=timezone.utc),
                    ),
                    Filter(
                        "date_time",
                        "<=",
                        datetime(2023, 1, 3, 0, 0, tzinfo=timezone.utc),
                    ),
                ],
                pl.DataFrame(
                    {
                        "implant_id": [1, 1, 2, 3, 3],
                        "date_time": [
                            datetime(2023, 1, 2, 0, 0, tzinfo=timezone.utc),
                            datetime(2023, 1, 3, 0, 0, tzinfo=timezone.utc),
                            datetime(2023, 1, 2, 8, 0, tzinfo=timezone.utc),
                            datetime(2023, 1, 1, 16, 0, tzinfo=timezone.utc),
                            datetime(2023, 1, 2, 16, 0, tzinfo=timezone.utc),
                        ],
                        "value": [20, 30, 50, 70, 80],
                    }
                ),
            ),
        ],
    )
    def test_read_partitioned(
        self,
        partitioned_parquet_table: ParquetTable,
        filters: InputFilters | None,
        expected: pl.DataFrame,
    ):
        result = partitioned_parquet_table(filters=filters).collect()

        assert (
            result.sort("value").select(expected.columns).equals(expected.sort("value"))
        )

    @pytest.mark.parametrize(
        ("filters", "expected"),
        [
            # No filters given
            (
                [],
                (
                    # Nothing is added to the URI
                    "",
                    # All partitions remain
                    [Partition("implant_id", pl.Int32), Partition("date", pl.String)],
                    # No filters are given to begin with
                    [],
                    # No filters are applied
                    [],
                ),
            ),
            # One partition filter given
            (
                [[Filter("implant_id", "=", 1)]],
                (
                    "/implant_id=1/",
                    [Partition("date", pl.String)],
                    [[]],
                    [Filter("implant_id", "=", 1)],
                ),
            ),
            # Two partition filters given
            (
                [[Filter("implant_id", "=", 1), Filter("date", "=", "2023-01-01")]],
                (
                    "/implant_id=1/date=2023-01-01/",
                    [],
                    # No filters are passed since all partitions are filtered out
                    [[]],
                    [Filter("implant_id", "=", 1), Filter("date", "=", "2023-01-01")],
                ),
            ),
            # Two partition filters + one non-partition filter given
            (
                [
                    [
                        Filter("implant_id", "=", 1),
                        Filter("date", "=", "2023-01-01"),
                        Filter("value", ">", 20),
                    ]
                ],
                (
                    "/implant_id=1/date=2023-01-01/",
                    [],
                    # The non-partition filter should still be passed
                    [[Filter("value", ">", 20)]],
                    [
                        Filter("implant_id", "=", 1),
                        Filter("date", "=", "2023-01-01"),
                    ],
                ),
            ),
            # Multiple sets of filters. They have different partition filters
            [
                [
                    [Filter("implant_id", "=", 1)],
                    [Filter("implant_id", "=", 2)],
                ],
                (
                    # Nothing should be applied. The / is an implementation detail
                    "/",
                    [
                        Partition("implant_id", pl.Int32),
                        Partition("date", pl.String),
                    ],
                    [
                        [Filter("implant_id", "=", 1)],
                        [Filter("implant_id", "=", 2)],
                    ],
                    [],
                ),
            ],
            # Multiple sets of filters, but all have the same implant_id filter
            # Since each filter set have different date partition filters
            # they are not applied to the URI
            (
                [
                    [Filter("implant_id", "=", 1), Filter("date", "=", "2023-01-01")],
                    # Order should not matter
                    [Filter("date", "=", "2023-01-02"), Filter("implant_id", "=", 1)],
                ],
                (
                    "/implant_id=1/",
                    [Partition("date", pl.String)],
                    [
                        [Filter("date", "=", "2023-01-01")],
                        [Filter("date", "=", "2023-01-02")],
                    ],
                    [Filter("implant_id", "=", 1)],
                ),
            ),
        ],
    )
    def test_build_uri_from_filters(
        self,
        partitioned_parquet_table: ParquetTable,
        filters: NormalizedFilters,
        expected: tuple[str, list[Partition], list[list[Filter]], list[Filter]],
    ):
        # NOTE: this is technically internal and very prone to changes, but we test it
        # to ensure we're not reading more partitions than we need to.
        uri, partitions, new_filters, applied_filters = (
            partitioned_parquet_table._build_uri_from_filters(filters=filters)
        )

        # We don't have access to tmp_path in given parameters, so just check suffix matches
        uri_suffix = uri.removeprefix(partitioned_parquet_table.uri)

        (
            expected_uri,
            expected_partitions,
            expected_filters,
            expected_applied_filters,
        ) = expected

        assert uri_suffix == expected_uri
        assert partitions == expected_partitions
        assert new_filters == expected_filters
        assert applied_filters == expected_applied_filters
