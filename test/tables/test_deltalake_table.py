import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from deltalake import DeltaTable, write_deltalake
import polars as pl
import pyarrow as pa
import pytest

from datarepo.core.tables.deltalake_table import (
    DeltaCacheOptions,
    DeltalakeTable,
    Filter,
    fetch_df_by_partition,
    fetch_dfs_by_paths,
)

test_schema = pa.schema(
    [
        ("implant_id", pa.int64()),
        ("date", pa.string()),
    ]
)


def pl_read_parquet(source, **kwargs):
    dataframes = {
        "df-empty.parquet": pl.DataFrame(),
        "df-ab.parquet": pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["test1", "test2", "test3"],
            }
        ),
        "df-ab2.parquet": pl.DataFrame(
            {
                "a": [4, 5, 6],
                "b": ["test4", "test5", "test6"],
            }
        ),
        "df-ab-reordered.parquet": pl.DataFrame(
            {
                "b": ["test7"],
                "a": [7],
            }
        ),
        "df-abc.parquet": pl.DataFrame(
            {
                "c": [6, 7],
                "a": [4, 5],
                "b": ["test4", "test5"],
            }
        ),
        "s3://test/table/implant_id=123/date=2024-01-01/df1.parquet": pl.DataFrame(
            {
                "implant_id": [123],
                "date": ["2024-01-01"],
                "value": [1],
            }
        ),
        "s3://test/table/implant_id=123/date=2024-01-01/df2.parquet": pl.DataFrame(
            {
                "implant_id": [123],
                "date": ["2024-01-01"],
                "value": [2],
            }
        ),
    }
    return dataframes[source]


@pytest.fixture(scope="session")
def mock_pl_read_parquet():
    with patch("polars.read_parquet", side_effect=pl_read_parquet):
        yield


@pytest.fixture(scope="session")
def mock_delta_rs_table():
    mock = MagicMock()
    mock.table_uri = "s3://test/table"
    mock.files.side_effect = files_patch
    return mock


def files_patch(partition_filters):
    if partition_filters == [("implant_id", "=", 123), ("date", "=", "2024-01-01")]:
        return [
            "implant_id=123/date=2024-01-01/df1.parquet",
            "implant_id=123/date=2024-01-01/df2.parquet",
        ]
    else:
        return []


@pytest.fixture
def delta_table_path(tmp_path: Path):
    return tmp_path / "test-delta-table"


@pytest.fixture
def delta_table_definition(delta_table_path: Path) -> DeltalakeTable:
    return DeltalakeTable(
        name="my_delta_table",
        schema=pa.schema(
            [
                ("implant_id", pa.int64()),
                ("date", pa.string()),
                ("uniq", pa.string()),
                ("value", pa.int64()),
            ]
        ),
        uri=str(delta_table_path),
        unique_columns=["uniq"],
    )


@pytest.fixture
def delta_table(delta_table_definition: DeltalakeTable) -> DeltaTable:
    return DeltaTable.create(
        table_uri=delta_table_definition.uri,
        schema=delta_table_definition.schema,
        partition_by=["implant_id", "date"],
    )


class TestDeltalakeTable:
    def test_fetch_dfs_by_paths(self, mock_pl_read_parquet):
        result = fetch_dfs_by_paths(
            ["df-ab.parquet", "df-ab2.parquet"],
            schema=pa.schema(
                [
                    ("a", pa.int64()),
                    ("b", pa.string()),
                ]
            ),
        )

        assert result.sort("a").equals(
            pl.DataFrame(
                {
                    "a": [1, 2, 3, 4, 5, 6],
                    "b": ["test1", "test2", "test3", "test4", "test5", "test6"],
                }
            )
        )

    def test_fetch_dfs_by_paths_reordered(self, mock_pl_read_parquet):
        result = fetch_dfs_by_paths(
            # One parquet has columns in a different order, but we should use schema's order
            ["df-ab.parquet", "df-ab-reordered.parquet"],
            schema=pa.schema(
                [
                    ("a", pa.int64()),
                    ("b", pa.string()),
                ]
            ),
        )

        assert result.sort("a").equals(
            pl.DataFrame(
                {
                    "a": [1, 2, 3, 7],
                    "b": ["test1", "test2", "test3", "test7"],
                }
            )
        )

    def test_fetch_dfs_by_paths_different_schemas(self, mock_pl_read_parquet):
        result = fetch_dfs_by_paths(
            ["df-ab.parquet", "df-abc.parquet"],
            schema=pa.schema(
                [
                    ("a", pa.int64()),
                    ("b", pa.string()),
                ]
            ),
        )

        # We only specified a and b in read schema, so c is dropped for df-abc
        assert result.sort("a").equals(
            pl.DataFrame(
                {
                    "a": [1, 2, 3, 4, 5],
                    "b": ["test1", "test2", "test3", "test4", "test5"],
                }
            )
        )

        result = fetch_dfs_by_paths(
            ["df-ab.parquet", "df-abc.parquet"],
            schema=pa.schema(
                [
                    ("a", pa.int64()),
                    ("b", pa.string()),
                    ("c", pa.int64()),
                ]
            ),
        )

        # c does not exist in first parquet, so is filled with nulls
        assert result.sort("a").equals(
            pl.DataFrame(
                {
                    "a": [1, 2, 3, 4, 5],
                    "b": ["test1", "test2", "test3", "test4", "test5"],
                    "c": [None] * 3 + [6, 7],
                }
            )
        )

    def test_fetch_df_by_partition(self, mock_delta_rs_table):
        result = fetch_df_by_partition(
            dt=mock_delta_rs_table,
            partition=[("implant_id", "=", 123), ("date", "=", "2024-01-01")],
            schema=pa.schema(
                [
                    ("implant_id", pa.int64()),
                    ("date", pa.string()),
                    ("value", pa.int64()),
                ]
            ),
        )

        assert result.sort("value").equals(
            pl.DataFrame(
                {
                    "implant_id": [123] * 2,
                    "date": ["2024-01-01"] * 2,
                    "value": [1, 2],
                }
            )
        )

    def test_fetch_df_by_partition_empty_files(self, mock_delta_rs_table):
        result = fetch_df_by_partition(
            dt=mock_delta_rs_table,
            partition=[("implant_id", "=", 123), ("date", "=", "2024-01-02")],
            schema=pa.schema(
                [
                    ("implant_id", pa.int64()),
                    ("date", pa.string()),
                    ("value", pa.int64()),
                ]
            ),
        )

        assert result.is_empty()
        assert list(result.columns) == ["implant_id", "date", "value"]

    @pytest.mark.parametrize(
        ("data", "args", "expected"),
        [
            # Test removing duplicates from the unique column
            (
                pl.DataFrame(
                    {
                        "implant_id": [5956, 5956, 5956, 5956, 5957],
                        "date": ["2024-01-01"] * 5,
                        "uniq": ["1", "1", "2", "3", "4"],
                        "value": [1, 1, 2, 3, 4],
                    }
                ),
                {"filters": [Filter("implant_id", "=", 5956)]},
                pl.DataFrame(
                    {
                        # Duplicates of the unique column uniq should be removed
                        "implant_id": [5956] * 3,
                        "date": ["2024-01-01"] * 3,
                        "uniq": ["1", "2", "3"],
                        "value": [1, 2, 3],
                    }
                ),
            ),
            # Same but with string filters
            (
                pl.DataFrame(
                    {
                        "implant_id": [5956, 5956, 5956, 5956, 5957],
                        "date": ["2024-01-01"] * 5,
                        "uniq": ["1", "1", "2", "3", "4"],
                        "value": [1, 1, 2, 3, 4],
                    }
                ),
                {"filters": "implant_id = 5956"},
                pl.DataFrame(
                    {
                        # Duplicates of the unique column z should be removed
                        "implant_id": [5956] * 3,
                        "date": ["2024-01-01"] * 3,
                        "uniq": ["1", "2", "3"],
                        "value": [1, 2, 3],
                    }
                ),
            ),
            # Multiple filter sets
            (
                pl.DataFrame(
                    {
                        "implant_id": [5956] * 5,
                        "date": [
                            "2024-04-05",
                            "2024-04-06",
                            "2024-04-07",
                            "2024-04-08",
                            "2024-04-09",
                        ],
                        "uniq": ["1", "2", "3", "4", "5"],
                        "value": [1, 2, 3, 4, 5],
                    },
                ),
                {
                    "filters": [
                        [
                            Filter("implant_id", "=", 5956),
                            Filter("date", "=", "2024-04-05"),
                        ],
                        [
                            Filter("implant_id", "=", 5956),
                            Filter("date", "=", "2024-04-06"),
                        ],
                        [
                            Filter("implant_id", "=", 5956),
                            Filter("date", "=", "2024-04-07"),
                        ],
                    ]
                },
                pl.DataFrame(
                    {
                        "implant_id": [5956] * 3,
                        "date": ["2024-04-05", "2024-04-06", "2024-04-07"],
                        "uniq": ["1", "2", "3"],
                        "value": [1, 2, 3],
                    }
                ),
            ),
            # Select subset of columns
            (
                pl.DataFrame(
                    {
                        "implant_id": [5956] * 4 + [5957],
                        "date": ["2024-01-01"] * 5,
                        "uniq": ["1", "1", "2", "3", "4"],
                        "value": [1, 1, 2, 3, 4],
                    }
                ),
                {
                    "filters": "implant_id = 5956",
                    # Unique columns should be used even when not selected
                    "columns": ["implant_id", "value"],
                },
                pl.DataFrame(
                    {
                        "implant_id": [5956] * 3,
                        "value": [1, 2, 3],
                    }
                ),
            ),
        ],
    )
    def test_call_loads_correct_data(
        self,
        data: pl.DataFrame,
        args: dict[str, Any],
        expected: pl.DataFrame,
        delta_table_definition: DeltalakeTable,
        delta_table: DeltaTable,
    ):
        write_deltalake(
            delta_table,
            data=data.to_arrow(),
            mode="overwrite",
        )

        actual_sorted = delta_table_definition(**args).collect().sort("value")
        expected_sorted = expected.sort("value")
        assert actual_sorted.equals(expected_sorted)

    def test_schema_version_uri_resolution(self):
        v1_schema = pa.schema([("id", pa.int64()), ("value", pa.int64())])
        v2_schema = pa.schema(
            [("id", pa.int64()), ("value", pa.int64()), ("note", pa.string())]
        )
        table = DeltalakeTable(
            name="signals",
            uri="s3://bucket/neural/signals",
            schema=v2_schema,
            schema_versions={"v1": v1_schema, "v2": v2_schema},
            default_schema_version="v2",
        )

        assert table.uri == "s3://bucket/neural/signals/v2"
        assert table.schema_version == "v2"
        assert table.at_version("v1").uri == "s3://bucket/neural/signals/v1"
        assert table.at_version("v1").schema.equals(v1_schema)

    def test_versioned_schema_reads(
        self,
        tmp_path: Path,
    ):
        base_path = tmp_path / "part"
        v1_schema = pa.schema(
            [
                ("p_partkey", pa.int64()),
                ("p_name", pa.string()),
                ("p_size", pa.int32()),
            ]
        )
        v2_schema = pa.schema(
            [
                ("p_partkey", pa.int64()),
                ("p_name", pa.string()),
                ("p_size", pa.int32()),
                ("p_comment", pa.string()),
            ]
        )

        v1_path = base_path / "v1"
        v2_path = base_path / "v2"

        v1_data = pl.DataFrame(
            {
                "p_partkey": [1, 2],
                "p_name": ["a", "b"],
                "p_size": [10, 20],
            }
        )
        v2_data = pl.DataFrame(
            {
                "p_partkey": [1, 2],
                "p_name": ["a", "b"],
                "p_size": [10, 20],
                "p_comment": ["old", "new"],
            }
        )

        v1_table = DeltaTable.create(table_uri=str(v1_path), schema=v1_schema)
        write_deltalake(v1_table, data=v1_data.to_arrow(), mode="overwrite")
        v2_table = DeltaTable.create(table_uri=str(v2_path), schema=v2_schema)
        write_deltalake(v2_table, data=v2_data.to_arrow(), mode="overwrite")

        part = DeltalakeTable(
            name="part",
            uri=str(base_path),
            schema=v2_schema,
            schema_versions={"v1": v1_schema, "v2": v2_schema},
            default_schema_version="v2",
        )

        v1_result = part.at_version("v1")().collect().sort("p_partkey")
        assert list(v1_result.columns) == ["p_partkey", "p_name", "p_size"]
        assert v1_result.equals(v1_data)

        v2_result = part().collect().sort("p_partkey")
        assert list(v2_result.columns) == [
            "p_partkey",
            "p_name",
            "p_size",
            "p_comment",
        ]
        assert v2_result.equals(v2_data)

        inline_v1 = part(schema_version="v1").collect().sort("p_partkey")
        assert inline_v1.equals(v1_data)

    def test_at_version_requires_schema_versions(self, delta_table_definition):
        with pytest.raises(ValueError, match="schema_versions"):
            delta_table_definition.at_version("v1")

    def test_unknown_schema_version_raises(self):
        schema = pa.schema([("id", pa.int64())])
        table = DeltalakeTable(
            name="t",
            uri="/tmp/t",
            schema=schema,
            schema_versions={"v1": schema},
            default_schema_version="v1",
        )
        with pytest.raises(KeyError, match="v9"):
            table.at_version("v9")

    """ this test is commented out until we upstream delta caching
    def test_delta_cache(
        self,
        delta_table_definition: DeltalakeTable,
        delta_table: DeltaTable,
        delta_table_path: Path,
        tmp_path: Path,
    ):
        write_deltalake(
            delta_table,
            data=pl.DataFrame(
                {
                    "implant_id": [1, 1, 1, 2, 2, 2],
                    "date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
                    "uniq": ["a", "b", "c", "d", "e", "f"],
                    "value": [1, 2, 3, 4, 5, 6],
                }
            ).to_arrow(),
            mode="overwrite",
        )

        # NOTE: the cache path behaves slightly differently when table is on S3
        # e.g. if table is s3://bucket/test-table on S3, the cache path will be
        # file_cache_path/test-table (instead of just file_cache_path)
        # This shouldn't matter for the purposes of this test
        # TODO(peter): this can be better covered by a full E2E test using minio
        base_cache_path = tmp_path / "delta-cache"
        assert base_cache_path != delta_table_path

        assert not os.path.exists(base_cache_path) or not os.listdir(base_cache_path)

        delta_table_definition(
            filters="implant_id = 1 and date >= '2024-01-02'",
            cache_options=DeltaCacheOptions(file_cache_path=base_cache_path),
        )

        cached_dirs = set(os.listdir(base_cache_path))
        assert "_delta_log" in cached_dirs
        assert "implant_id=1" in cached_dirs

        assert os.listdir(base_cache_path / "implant_id=1" / "date=2024-01-02")
        assert os.listdir(base_cache_path / "implant_id=1" / "date=2024-01-03")
    """
