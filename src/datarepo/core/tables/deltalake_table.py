# mypy: disable-error-code=override
from __future__ import annotations

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Any, TypeAlias
import warnings

import boto3
from deltalake import DeltaTable, QueryBuilder
from deltalake.warnings import ExperimentalWarning
import polars as pl
import pyarrow as pa

from datarepo.core.dataframe import NlkDataFrame
from datarepo.core.tables.filters import InputFilters, normalize_filters
from datarepo.core.tables.metadata import (
    TableMetadata,
    TableProtocol,
    TableSchema,
    TableColumn,
    TablePartition,
)
from datarepo.core.tables.util import (
    DeltaRoapiOptions,
    Filter,
    RoapiOptions,
    filters_to_sql_predicate,
    get_storage_options,
)

READ_PARQUET_RETRY_COUNT = 10
DEFAULT_TIMEOUT = "150s"

DeltaInputFilters: TypeAlias = InputFilters | str


@dataclass
class DeltaCacheOptions:
    # Path to the directory where files are cached. This can be the same for all tables
    # since the S3 table prefix is included in the cached paths.
    file_cache_path: str
    # Duration for which the _last_checkpoint file is cached. A longer duration means
    # we can reuse cached checkpoints for longer, which could improve loading performance
    # from not downloading new checkpoint parquets. However, we also need to load more
    # individual transaction jsons, so this should not be set too high.
    # A reasonable value for a table with frequent updates (e.g. binned spikes) is 30m.
    file_cache_last_checkpoint_valid_duration: str | None = None

    def to_storage_options(self) -> dict[str, Any]:
        """Convert the cache options to a dictionary of storage options.

        Returns:
            dict[str, Any]: A dictionary of storage options that can be used with DeltaTable.
        """
        opts = {
            "file_cache_path": os.path.expanduser(self.file_cache_path),
        }
        if self.file_cache_last_checkpoint_valid_duration is not None:
            opts["file_cache_last_checkpoint_valid_duration"] = (
                self.file_cache_last_checkpoint_valid_duration
            )
        return opts


class DeltalakeTable(TableProtocol):
    """A table that is backed by a Delta Lake table."""

    def __init__(
        self,
        name: str,
        uri: str,
        schema: pa.Schema,
        description: str = "",
        docs_filters: list[Filter] = [],
        docs_columns: list[str] | None = None,
        roapi_opts: RoapiOptions | None = None,
        unique_columns: list[str] | None = None,
        table_metadata_args: dict[str, Any] | None = None,
        stats_cols: list[str] | None = None,
        extra_cols: list[tuple[pl.Expr, str]] | None = None,
    ):
        """Initialize the DeltalakeTable.

        Args:
            name (str): table name, used as the table identifier in the DeltaTable
            uri (str): uri of the table, e.g. "s3://bucket/path/to/table"
            schema (pa.Schema): schema of the table, used to define the table structure
            description (str, optional): description of the table, used for documentation. Defaults to "".
            docs_filters (list[Filter], optional): documentation filters, used to filter the table in the documentation. Defaults to [].
            docs_columns (list[str] | None, optional): documentation columns, used to define the columns in the documentation. Defaults to None.
            roapi_opts (RoapiOptions | None, optional): ROAPI options, used to configure the ROAPI for the table. Defaults to None.
            unique_columns (list[str] | None, optional): unique columns in the table, used to optimize the read performance. Defaults to None.
            table_metadata_args (dict[str, Any] | None, optional): table metadata arguments, used to configure the table metadata. Defaults to None.
            stats_cols (list[str] | None, optional): statistics columns, used to define the columns that have statistics. Defaults to None.
            extra_cols (list[tuple[pl.Expr, str]] | None, optional): extra columns to add to the table, where each tuple contains a Polars expression and its type annotation. Defaults to None.
        """
        self.name = name
        self.uri = uri
        self.schema = schema
        self.unique_columns = unique_columns
        self.stats_cols = stats_cols or []
        self.extra_cols = extra_cols or []

        self.table_metadata = TableMetadata(
            table_type="DELTA_LAKE",
            description=description,
            docs_args={"filters": docs_filters, "columns": docs_columns},
            roapi_opts=roapi_opts or DeltaRoapiOptions(),
            **(table_metadata_args or {}),
        )

    def get_schema(self) -> TableSchema:
        """Generate and return the schema of the table, including partitions and columns.

        Returns:
            TableSchema: table schema containing partition and column information.
        """
        dt = self.delta_table()
        schema = self.schema
        partition_cols = dt.metadata().partition_columns
        filters = {
            f.column: f.value
            for f in self.table_metadata.docs_args.get("filters", [])
            if isinstance(f, Filter)
        }
        partitions = [
            TablePartition(
                column_name=col,
                type_annotation=str(schema.field(col).type),
                value=filters.get(col),
            )
            for col in partition_cols
        ]
        columns = [
            TableColumn(
                column=name,
                type=str(schema.field(name).type),
                readonly=False,
                filter_only=False,
                has_stats=name in partition_cols or name in self.stats_cols,
            )
            for name in schema.names
        ]
        columns += [
            TableColumn(
                column=expr.meta.output_name(),
                type=expr_type,
                readonly=True,
                filter_only=False,
                has_stats=False,
            )
            for expr, expr_type in self.extra_cols
        ]
        return TableSchema(
            partitions=partitions,
            columns=columns,
        )

    def __call__(
        self,
        filters: DeltaInputFilters | None = None,
        columns: list[str] | None = None,
        boto3_session: boto3.Session | None = None,
        endpoint_url: str | None = None,
        timeout: str | None = None,
        cache_options: DeltaCacheOptions | None = None,
        **kwargs: Any,
    ) -> NlkDataFrame:
        """Fetch a dataframe from the Delta Lake table.

        Args:
            filters (DeltaInputFilters | None, optional): filters to apply to the table. Defaults to None.
            columns (list[str] | None, optional): columns to select from the table. Defaults to None.
            boto3_session (boto3.Session | None, optional): boto3 session to use for S3 access. Defaults to None.
            endpoint_url (str | None, optional): endpoint URL for S3 access. Defaults to None.
            timeout (str | None, optional): timeout for S3 access. Defaults to None.
            cache_options (DeltaCacheOptions | None, optional): cache options for the Delta Lake table. Defaults to None.

        Returns:
            NlkDataFrame: a dataframe containing the data from the Delta Lake table, filtered and selected according to the provided parameters.
        """
        storage_options = {
            "timeout": timeout or DEFAULT_TIMEOUT,
            **get_storage_options(
                boto3_session=boto3_session, endpoint_url=endpoint_url
            ),
        }
        if cache_options is not None:
            storage_options = {
                **storage_options,
                **cache_options.to_storage_options(),
            }
        dt = self.delta_table(storage_options=storage_options)

        return self.construct_df(dt=dt, filters=filters, columns=columns)

    def construct_df(
        self,
        dt: DeltaTable,
        filters: DeltaInputFilters | None = None,
        columns: list[str] | None = None,
    ) -> NlkDataFrame:
        """Construct a dataframe from the Delta Lake table.

        Args:
            dt (DeltaTable): The DeltaTable object representing the Delta Lake table.
            filters (DeltaInputFilters | None, optional): filters to apply to the table. Defaults to None.
            columns (list[str] | None, optional): columns to select from the table. Defaults to None.

        Returns:
            NlkDataFrame: a dataframe containing the data from the Delta Lake table, filtered and selected according to the provided parameters.
        """
        # Use schema defined on this table, the physical schema in deltalake metadata might be different
        schema = self.schema

        predicate_str = datafusion_predicate_from_filters(schema, filters)

        # These should not be read because they don't exist in the delta table
        extra_col_exprs = [expr for expr, _ in self.extra_cols]
        extra_column_names = set(expr.meta.output_name() for expr in extra_col_exprs)

        columns_to_read = None
        unique_column_names = set(self.unique_columns or [])
        if columns:
            columns_to_read = list(
                (set(columns) | unique_column_names) - extra_column_names
            )

        # TODO(peter): consider a sql builder for more complex queries?
        select_cols = (
            ", ".join([f'"{col}"' for col in columns_to_read])
            if columns_to_read
            else "*"
        )
        condition = f"WHERE {predicate_str}" if predicate_str else ""
        query_string = f"""
            SELECT {select_cols}
            FROM "{self.name}"
            {condition}
        """
        with warnings.catch_warnings():
            # Ignore ExperimentalWarning emitted from QueryBuilder
            warnings.filterwarnings("ignore", category=ExperimentalWarning)
            batches = (
                QueryBuilder().register(self.name, dt).execute(query_string).fetchall()
            )

        # Since we might cast unique string columns to categoricals, use a string cache to
        # improve performance when combining multiple dataframes
        with pl.StringCache():
            if batches:
                frame = pl.from_arrow(batches, rechunk=False)
                if isinstance(frame, pl.Series):
                    frame = frame.to_frame()
                frame = _normalize_df(frame, self.schema, columns=columns_to_read)
            else:
                # If dataset is empty, the returned dataframe will have no columns
                frame = _empty_normalized_df(schema)

            if self.extra_cols:
                frame = frame.with_columns(extra_col_exprs)

            if self.unique_columns:
                # Cast unique string columns to categoricals first
                # In some cases, this reduces peak memory usage up to 50%
                curr_schema = frame.schema
                cat_schema = {
                    col: pl.Categorical
                    for col in curr_schema
                    if curr_schema[col] == pl.String
                }
                frame = (
                    frame.cast(cat_schema)  # type: ignore[arg-type]
                    .unique(subset=self.unique_columns, maintain_order=True)
                    .cast(curr_schema)  # type: ignore[arg-type]
                )

        if columns:
            frame = frame.select(columns)

        return frame.lazy()

    def delta_table(
        self, storage_options: dict[str, Any] | None = None, version: int | None = None
    ) -> DeltaTable:
        """Get the DeltaTable object for this table.

        Args:
            storage_options (dict[str, Any] | None, optional): Storage options for the DeltaTable, such as S3 access credentials. Defaults to None.
            version (int | None, optional): Version of the Delta table to read. If None, the latest version is used. Defaults to None.

        Returns:
            DeltaTable: The DeltaTable object representing the Delta Lake table.
        """
        return DeltaTable(
            table_uri=self.uri, storage_options=storage_options, version=version
        )


def fetch_df_by_partition(
    dt: DeltaTable,
    partition: list[tuple[str, str, Any]],
    schema: pa.Schema,
    storage_options: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """
    The native delta-rs read has slower performance. The difference comes from the dataset in
    dt.to_pyarrow_dataset() - this dataset is slower than the one manually constructed below.

    pyarrow.dataset.Dataset.to_table() reading was also tried, however this also resulted
    in slower performance for larger datasets. For example, reading P1 webgrid data on 2024-05-21
    (2.1M+ rows) takes ~20+ seconds.

    The hypothesized reason(s) for the slow read performance of pyarrow.dataset.Dataset.to_table()
    are
    1. Additional step of metadata loading
    2. Lazy loading which could lead to longer load times
    Source: https://github.com/apache/arrow/issues/35332

    When using pyarrow's parquet read_table(), which will eagerly read the entire dataset, the
    read for the P1 webgrid data on 2024-05-21 (2.1M+ rows) takes ~7-10 seconds.
    Thus, it makes sense to use read_table() since we have a predefined list of files to be loaded
    from dt.files

    """
    files = dt.files(partition_filters=partition)

    files = [os.path.join(dt.table_uri, file) for file in files]

    if not files:
        # NOTE: polars has a bug where with_columns(...) on an empty dataframe with no columns will add a row
        # for all new columns. Workaround by adding the columns before normalizing
        return _empty_normalized_df(schema)

    return fetch_dfs_by_paths(
        files=files, schema=schema, storage_options=storage_options
    )


def fetch_dfs_by_paths(
    files: list[str],
    schema: pa.Schema,
    storage_options: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Fetch dataframes from a list of Parquet files using multithreading.

    Args:
        files (list[str]): List of file paths to read Parquet files from.
        schema (pa.Schema): Schema to normalize the dataframes to.
        storage_options (dict[str, Any] | None, optional): Storage options for reading the Parquet files, such as S3 access credentials. Defaults to None.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the concatenated results of all Parquet files, normalized to the specified schema.
    """
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                pl.read_parquet,
                file,
                retries=READ_PARQUET_RETRY_COUNT,
                storage_options=storage_options,
                # Hive partitioning was disabled by default in https://github.com/pola-rs/polars/pull/17106
                hive_partitioning=True,
            )
            for file in files
        ]
        concurrent.futures.wait(futures)

    dfs = [future.result() for future in futures]

    return pl.concat([_normalize_df(df, schema=schema) for df in dfs])


def _empty_normalized_df(schema: pa.Schema) -> pl.DataFrame:
    """Create an empty DataFrame with the specified schema.

    Args:
        schema (pa.Schema): The schema to use for the empty DataFrame.

    Returns:
        pl.DataFrame: An empty DataFrame with the specified schema.
    """
    return _normalize_df(pl.DataFrame({col: [] for col in schema.names}), schema=schema)


def _normalize_df(
    df: pl.DataFrame,
    schema: pa.Schema,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Add missing columns, cast, and reorder dataframe columns to the specified schema's order.

    If columns is provided, only those columns will be added.

    Args:
        df (pl.DataFrame): dataframe to normalize.
        schema (pa.Schema): schema to normalize the dataframe to.
        columns (list[str] | None, optional): List of columns to include in the normalized dataframe. If None, all columns from the schema are included. Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame normalized to the specified schema, with missing columns added and columns reordered.
    """
    empty_frame = pl.from_arrow(schema.empty_table())
    if isinstance(empty_frame, pl.Series):
        empty_frame = empty_frame.to_frame()
    polars_schema = empty_frame.schema
    if columns:
        # Only add back specified columns
        polars_schema = pl.Schema(
            {col: dtype for col, dtype in polars_schema.items() if col in columns}
        )

    schema_columns = list(polars_schema.keys())
    missing_columns = set(schema_columns) - set(df.columns)
    return (
        df.with_columns(pl.lit(None).alias(col) for col in missing_columns)
        .with_columns([pl.col(col).cast(dtype) for col, dtype in polars_schema.items()])
        .select(schema_columns)
    )


def datafusion_predicate_from_filters(
    schema: pa.Schema, filters: DeltaInputFilters | None
) -> str | None:
    """Convert input filters to a SQL predicate string for use in Delta Lake queries.

    Args:
        schema (pa.Schema): The schema of the Delta Lake table, used to validate column names and types.
        filters (DeltaInputFilters | None): The filters to apply to the Delta Lake table. This can be a string or a list of Filter objects.

    Returns:
        str | None: A SQL predicate string that can be used in Delta Lake queries, or None if no filters are provided.
    """
    if not filters:
        return None
    elif isinstance(filters, str):
        return filters

    normalized_filters = normalize_filters(filters)
    return filters_to_sql_predicate(schema, normalized_filters)
