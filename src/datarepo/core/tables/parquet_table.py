# mypy: disable-error-code=override

from os import path
from typing import Any, Optional, Sequence

import boto3
import polars as pl

from datarepo.core.dataframe import NlkDataFrame
from datarepo.core.tables.filters import (
    InputFilters,
    NormalizedFilters,
    normalize_filters,
)
from datarepo.core.tables.metadata import (
    TableMetadata,
    TableProtocol,
    TableSchema,
    TablePartition,
    TableColumn,
)
from datarepo.core.tables.util import (
    Filter,
    Partition,
    PartitioningScheme,
    RoapiOptions,
    exactly_one_equality_filter,
    get_storage_options,
)


def pl_all(exprs: Sequence[pl.Expr]) -> pl.Expr:
    """Combine a sequence of polars expressions with logical AND.

    Args:
        exprs (Sequence[pl.Expr]): A sequence of polars expressions to combine.

    Returns:
        pl.Expr: A single polars expression that represents the logical AND of all input expressions.
    """
    assert len(exprs) > 0

    result = exprs[0]
    for expr in exprs[1:]:
        result = result & expr

    return result


def pl_any(exprs: Sequence[pl.Expr]) -> pl.Expr:
    """Combine a sequence of polars expressions with logical OR.

    Args:
        exprs (Sequence[pl.Expr]): A sequence of polars expressions to combine.

    Returns:
        pl.Expr: A single polars expression that represents the logical OR of all input expressions.
    """
    assert len(exprs) > 0

    result = exprs[0]
    for expr in exprs[1:]:
        result = result | expr

    return result


def _filters_to_conjunction_expr(filters: list[Filter]) -> pl.Expr | None:
    """For a list of filters, return a polars expression that represents
    the conjunction (AND) of all filters.

    Args:
        filters (list[Filter]): A list of filters to combine.

    Returns:
        pl.Expr | None: A polars expression that represents the conjunction of all filters,
        or None if the list is empty.
    """
    if not filters:
        return None

    result = _filter_to_expr(filters[0])
    for filter in filters[1:]:
        result = result & _filter_to_expr(filter)

    return pl_all([_filter_to_expr(filter) for filter in filters])


def _filters_to_expr(filters: NormalizedFilters) -> pl.Expr | None:
    """For a list of lists of filters, return a polars expression that represents
    the disjunction (OR) of all conjunctions of filters.

    Args:
        filters (NormalizedFilters): A list of lists of filters, where each inner list represents a conjunction (AND) of filters.

    Returns:
        pl.Expr | None: A polars expression that represents the disjunction of all conjunctions of filters,
        or None if there are no filters.
    """
    conjunctions = [_filters_to_conjunction_expr(filter_set) for filter_set in filters]
    not_none_conjunctions = [c for c in conjunctions if c is not None]
    if not not_none_conjunctions:
        return None

    return pl_any(not_none_conjunctions)


def _filter_to_expr(filter: Filter) -> pl.Expr:
    """Convert a single filter to a polars expression.

    Args:
        filter (Filter): A filter object containing the column, operator, and value.

    Raises:
        ValueError: If the operator is not supported.

    Returns:
        pl.Expr: A polars expression that represents the filter condition.
    """
    if filter.operator == "=":
        return pl.col(filter.column) == filter.value
    elif filter.operator == "!=":
        return pl.col(filter.column) != filter.value
    elif filter.operator == "<":
        return pl.col(filter.column) < filter.value
    elif filter.operator == ">":
        return pl.col(filter.column) > filter.value
    elif filter.operator == "<=":
        return pl.col(filter.column) <= filter.value
    elif filter.operator == ">=":
        return pl.col(filter.column) >= filter.value
    elif filter.operator == "in":
        return pl.col(filter.column).is_in(filter.value)
    elif filter.operator == "not in":
        return ~pl.col(filter.column).is_in(filter.value)
    elif filter.operator == "contains":
        return pl.col(filter.column).str.contains(filter.value)
    elif filter.operator == "includes":
        return pl.col(filter.column).list.contains(filter.value)
    elif filter.operator == "includes any":
        return pl_any(
            [pl.col(filter.column).list.contains(value) for value in filter.value]
        )
    elif filter.operator == "includes all":
        return pl_all(
            [pl.col(filter.column).list.contains(value) for value in filter.value]
        )
    else:
        raise ValueError(f"Unsupported operator {filter.operator}")


class ParquetTable(TableProtocol):
    """A table that is stored in Parquet format."""

    def __init__(
        self,
        name: str,
        uri: str,
        partitioning: list[Partition],
        partitioning_scheme: PartitioningScheme = PartitioningScheme.DIRECTORY,
        description: str = "",
        docs_filters: list[Filter] = [],
        docs_columns: list[str] | None = None,
        roapi_opts: RoapiOptions | None = None,
        parquet_file_name: str = "df.parquet",
        table_metadata_args: dict[str, Any] | None = None,
    ):
        """Initialize the ParquetTable.

        Args:
            name (str): name of the table, used for documentation and metadata.
            uri (str): uri of the table, typically an S3 bucket path.
            partitioning (list[Partition]): partitioning scheme for the table.
                This is a list of Partition objects, which define the columns and types used for partitioning
            partitioning_scheme (PartitioningScheme, optional): scheme used for partitioning.
                Defaults to PartitioningScheme.DIRECTORY.
            description (str, optional): description of the table, used for documentation.
                Defaults to "".
            docs_filters (list[Filter], optional): documentation filters for the table.
                These filters are used to generate documentation and are not applied to the data.
                Defaults to [].
            docs_columns (list[str] | None, optional): docsumentation columns for the table.
                These columns are used to generate documentation and are not applied to the data.
                Defaults to None.
            roapi_opts (RoapiOptions | None, optional): Read-only API options for the table.
                These options are used to configure the ROAPI endpoint for the table.
                Defaults to None.
            parquet_file_name (str, optional): parquet file name to use when building file fragments.
            table_metadata_args (dict[str, Any] | None, optional): additional metadata arguments for the table.

        Raises:
            ValueError: if the partitioning_scheme is not a valid PartitioningScheme.
        """
        if not isinstance(partitioning_scheme, PartitioningScheme):
            raise ValueError(f"Invalid partitioning scheme, got {partitioning_scheme}")

        self.name = name
        self.uri = uri
        self.partitioning = partitioning
        self.partitioning_scheme = partitioning_scheme

        self.table_metadata = TableMetadata(
            table_type="PARQUET",
            description=description,
            docs_args={"filters": docs_filters, "columns": docs_columns},
            roapi_opts=roapi_opts,
            **(table_metadata_args or {}),
        )

        self.parquet_file_name = parquet_file_name

    def get_schema(self) -> TableSchema:
        """Generates the schema of the table, including partitions and columns.

        Returns:
            TableSchema: table schema containing partitions and columns.
        """
        partitions = [
            TablePartition(
                column_name=filter.column,
                type_annotation=type(filter.value).__name__,
                value=filter.value,
            )
            for filter in self.table_metadata.docs_args.get("filters", [])
        ]

        # Pop the selected columns so that we still pull the full schema below
        docs_args = {**self.table_metadata.docs_args}
        docs_args.pop("columns")

        columns = []
        if docs_args or not partitions:
            table: NlkDataFrame = self(**docs_args)
            columns = [
                TableColumn(
                    column=key,
                    type=type.__str__(),
                    readonly=False,
                    filter_only=False,
                    has_stats=False,
                )
                for key, type in table.schema.items()
            ]

        return TableSchema(partitions=partitions, columns=columns)

    def __call__(
        self,
        filters: InputFilters | None = None,
        columns: Optional[list[str]] = None,
        boto3_session: boto3.Session | None = None,
        endpoint_url: str | None = None,
        **kwargs: Any,
    ) -> NlkDataFrame:
        """Fetches data from the Parquet table based on the provided filters and columns.

        Args:
            filters (InputFilters | None, optional): filters to apply to the data. Defaults to None.
            columns (Optional[list[str]], optional): columns to select from the data. Defaults to None.
            boto3_session (boto3.Session | None, optional): boto3 session to use for S3 access. Defaults to None.
            endpoint_url (str | None, optional): endpoint URL for S3 access. Defaults to None.

        Returns:
            NlkDataFrame: A DataFrame containing the filtered data from the Parquet table.
        """
        normalized_filters = normalize_filters(filters)
        (
            uri,
            remaining_partitions,
            remaining_filters,
            applied_filters,
        ) = self._build_uri_from_filters(normalized_filters)

        storage_options = get_storage_options(
            boto3_session=boto3_session,
            endpoint_url=endpoint_url,
        )

        df = pl.scan_parquet(
            uri,
            hive_partitioning=len(remaining_partitions) > 0,
            hive_schema={
                partition.column: partition.col_type
                for partition in remaining_partitions
            },
            allow_missing_columns=True,
            storage_options=storage_options,
        )

        if applied_filters:
            # Add columns removed from partitions and added to uri
            df = df.with_columns(
                pl.lit(f.value)
                .cast(
                    next(
                        partition.col_type
                        for partition in self.partitioning
                        if partition.column == f.column
                    )
                )
                .alias(f.column)
                for f in applied_filters
            )

        if remaining_filters:
            filter_expr = _filters_to_expr(remaining_filters)
            if filter_expr is not None:
                df = df.filter(filter_expr)

        if columns:
            df = df.select(columns)

        return df

    def build_file_fragment(self, filters: list[Filter]) -> str:
        """
        Returns a file path from the base table URI with the given filters.
        This will raise an error if the filter does not specify all partitions.

        This is currently used to generate the file path used by ROAPI to infer schemas.
        """
        uri_with_prefix, partitions, _, _ = self._build_uri_from_filters(
            normalize_filters(filters), include_base_uri=False
        )
        if len(partitions) > 0:
            partition_names = [partition.column for partition in partitions]
            raise ValueError(
                f"Not enough partitions specified, missing: {partition_names}"
            )

        return path.join(uri_with_prefix, self.parquet_file_name)

    def _build_uri_from_filters(
        self,
        filters: NormalizedFilters,
        include_base_uri: bool = True,
    ) -> tuple[str, list[Partition], NormalizedFilters, list[Filter]]:
        """Attempts to build an S3 list prefix from the given filters.
        We do this because pyarrow runs an S3 List() query on the base URI
        before applying filters to possible files. This can be slow.

        We can force pyarrow to do more sophisticated prefix filtering
        if we pre-construct the URI before making the call. If a partition
        has exactly one filter that uses strict equality, we know that
        it will contain that filter string in the URI. We attempt to do this
        for each partition filter, in order, until we encounter a
        partition that:
            1. does not have a filter, or
            2. has a more than one filters, or
            3. has filters that are not strict equality checks.

        This gives us the longest URL that can be used as a prefix filter for
        all files returned by the S3 List() call.

        In a benchmark, this brought reading ~1 million rows of binned spike files
        from 12s to 1.5s.

        In the future, we can further optimize this by building a list of
        candidate URIs and doing separate prefix filtering with each of those, in parallel.

        Long-term we should push this logic into pyarrow and make an upstream commit.

        NOTE: Add trailing slash - this is important to ensure that,
        when partitioning only by implant ID, a future 5-digit implant beginning with the same 4 digits
        as a 4-digit implant is not included in a 4-digit implant's query.
        """
        uri = self.uri if include_base_uri else ""

        if not filters or not self.partitioning:
            return uri, self.partitioning, filters, []

        partitions = list(self.partitioning)
        filters = [list(filter_set) for filter_set in filters]
        applied_filters = []

        for i, partition in enumerate(self.partitioning):
            partition_filters = [
                exactly_one_equality_filter(partition, f) for f in filters
            ]

            if len(partition_filters) == 0:
                break

            # either 0 or multiple filters for this partition,
            # break and deal with the s3 list() query
            if any(partition_filter is None for partition_filter in partition_filters):
                break

            # Only move forward if all remaining filter sets have the same partition filter
            if not all(
                partition_filter == partition_filters[0]
                for partition_filter in partition_filters
            ):
                break

            partition_filter = partition_filters[0]

            if partition_filter is None:
                continue

            if self.partitioning_scheme == PartitioningScheme.DIRECTORY:
                partition_component = str(partition_filter.value)
            elif self.partitioning_scheme == PartitioningScheme.HIVE:
                partition_component = (
                    partition.column + "=" + str(partition_filter.value)
                )

            uri = path.join(uri, partition_component)

            # remove the partition and filter since it's been applied
            # technically might not need to remove the partitions,
            # but they are semantically meaningless as we already have
            # constructed the URI, so we can pop them
            partitions.remove(partition)
            for filter_set in filters:
                filter_set.remove(partition_filter)

            applied_filters.append(partition_filter)

        uri = path.join(
            uri, ""
        )  # trailing slash prevents inclusion of partitions that are subsets of other partitions

        return (uri, partitions, filters, applied_filters)
