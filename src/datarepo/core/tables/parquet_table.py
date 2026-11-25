# mypy: disable-error-code=override

import datetime
import re
from os import path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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


_FIXED_OFFSET_RE = re.compile(r"^(?P<sign>[+-])(?P<hours>\d{2}):(?P<minutes>\d{2})$")


def _parse_iso_datetime(value: str) -> datetime.datetime:
    """Parse an ISO 8601 datetime string the same way on every supported Python.

    ``datetime.fromisoformat`` only accepts a trailing ``Z`` (UTC) suffix on
    Python 3.11+; on the 3.10 floor this project supports (``pyproject.toml``
    sets ``requires-python = ">=3.10"``), ``"2024-01-01T13:00:00Z"`` raises
    ``ValueError`` (confirmed on 3.10.12), while the exact same string parses
    on 3.12. Normalize the suffix first so both floors accept it. A date-only
    string (no time part, e.g. ``"2024-01-01"``) is accepted by
    ``fromisoformat`` as midnight on every supported version; no normalization
    is needed for that case.
    """
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    return datetime.datetime.fromisoformat(value)


def _resolve_timezone(tz_str: str) -> datetime.tzinfo:
    """Resolve a polars Datetime dtype's ``time_zone`` string to a ``tzinfo``.

    Most columns carry an IANA zone name (``"UTC"``, ``"America/New_York"``),
    which ``ZoneInfo`` resolves directly. But a Parquet timestamp's zone can
    also be stored as a bare fixed UTC offset (``"-05:00"``, ``"+05:30"``):
    confirmed by writing a `pa.timestamp("us", tz="-05:00")` column to Parquet
    and reading it back with ``pl.scan_parquet`` for a non-zero offset
    (``"+00:00"`` itself is normalized to ``"UTC"`` by polars, but other
    offsets are not). ``ZoneInfo`` has no IANA key for those and raises
    ``ZoneInfoNotFoundError``; parse the offset directly instead.
    """
    try:
        return ZoneInfo(tz_str)
    except ZoneInfoNotFoundError:
        match = _FIXED_OFFSET_RE.match(tz_str)
        if match is None:
            raise
        sign = 1 if match["sign"] == "+" else -1
        hours, minutes = int(match["hours"]), int(match["minutes"])
        if minutes == 0:
            # A whole-hour offset like this is itself renamed to the matching
            # Etc/GMT zone once a LazyFrame scanning such a column is
            # materialized: confirmed by round-tripping a
            # `pa.timestamp(..., tz="-05:00")` column through Parquet, where
            # `pl.scan_parquet(...).collect_schema()` reports "-05:00" but
            # `.collect()` reports "Etc/GMT+5". A plain `datetime.timezone`
            # offset here matches the pre-collect schema but not the
            # materialized one, and polars raises `ComputeError: unable to
            # parse time zone: 'UTC-05:00'` once the filter actually runs
            # against the real data (confirmed with the same repro). Use the
            # same Etc/GMT zone polars will materialize so the literal
            # doesn't disagree with it. Etc/GMT's sign convention is inverted
            # relative to ISO offsets (Etc/GMT+5 is UTC-05:00).
            return ZoneInfo(f"Etc/GMT{'-' if sign > 0 else '+'}{hours}")
        # A non-whole-hour fixed offset (e.g. "+05:30") has no Etc/GMT
        # equivalent, but separately confirmed that polars can't materialize
        # such a column via pl.scan_parquet at all in this version (a plain,
        # unfiltered `.collect()` already raises the same `ComputeError`), so
        # this table can't be read regardless of the filter. Fall back to a
        # plain offset so at least building the expression here doesn't add
        # its own `ZoneInfoNotFoundError` on top of that pre-existing limit.
        offset = datetime.timedelta(hours=hours, minutes=minutes)
        return datetime.timezone(sign * offset)


def _coerce_temporal_value(value: Any, dtype: pl.DataType | None) -> Any:
    """Parse an ISO date/datetime string so it can be compared to a Date/Datetime column.

    DataFusion (used for ``DeltalakeTable`` filters, see ``value_to_sql_expr`` in
    util.py) casts a quoted date/timestamp string literal to the column's type,
    "applying any UTC offset in the value" (see the comment there) rather than
    keeping a mismatched timezone. Polars does neither on its own: comparing a
    Date/Datetime column directly to a plain string raises
    ``InvalidOperationError``, and a naive `pl.lit` cast to a tz-aware
    Datetime column (or vice versa) raises ``SchemaError`` even once parsed, a
    different failure for the same backend mismatch.
    ``pl.lit(value).cast(dtype)`` was tried first and rejected: polars' str ->
    Datetime cast fails outright, even for a value that matches the column's own
    format (confirmed empirically), so it can't replace the parsing done here.
    Mirror DataFusion's behavior with plain datetime/zoneinfo arithmetic instead:
    parse the string, then apply the column's timezone the same way DataFusion
    does, producing a value polars can compare without a dtype mismatch.
    """
    if not isinstance(value, str):
        return value
    if dtype == pl.Date:
        try:
            return datetime.date.fromisoformat(value)
        except ValueError as e:
            raise ValueError(
                f"Could not parse {value!r} as a date (expected ISO 8601, "
                f"e.g. '2024-01-01'): {e}"
            ) from e
    elif isinstance(dtype, pl.Datetime):
        try:
            parsed = _parse_iso_datetime(value)
        except ValueError as e:
            raise ValueError(
                f"Could not parse {value!r} as a datetime (expected ISO 8601, "
                f"e.g. '2024-01-01 13:00:00', '2024-01-01 13:00:00+00:00', or "
                f"'2024-01-01T13:00:00Z'): {e}"
            ) from e
        if dtype.time_zone is None:
            if parsed.tzinfo is not None:
                # Apply the value's UTC offset instead of leaving a tz-aware
                # literal that can't be compared to this naive column.
                parsed = parsed.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        else:
            target_tz = _resolve_timezone(dtype.time_zone)
            if parsed.tzinfo is None:
                # No offset in the value: treat it as already being in the
                # column's timezone, rather than guessing it means UTC.
                parsed = parsed.replace(tzinfo=target_tz)
            else:
                parsed = parsed.astimezone(target_tz)
        return parsed
    return value


def _filters_to_conjunction_expr(
    filters: list[Filter], schema: pl.Schema | None = None
) -> pl.Expr | None:
    """For a list of filters, return a polars expression that represents
    the conjunction (AND) of all filters.

    Args:
        filters (list[Filter]): A list of filters to combine.
        schema (pl.Schema | None, optional): schema of the table, used to coerce
            string filter values for temporal columns. Defaults to None.

    Returns:
        pl.Expr | None: A polars expression that represents the conjunction of all filters,
        or None if the list is empty.
    """
    if not filters:
        return None

    return pl_all([_filter_to_expr(filter, schema) for filter in filters])


def _filters_to_expr(
    filters: NormalizedFilters, schema: pl.Schema | None = None
) -> pl.Expr | None:
    """For a list of lists of filters, return a polars expression that represents
    the disjunction (OR) of all conjunctions of filters.

    Args:
        filters (NormalizedFilters): A list of lists of filters, where each inner list represents a conjunction (AND) of filters.
        schema (pl.Schema | None, optional): schema of the table, used to coerce
            string filter values for temporal columns. Defaults to None.

    Returns:
        pl.Expr | None: A polars expression that represents the disjunction of all conjunctions of filters,
        or None if there are no filters.
    """
    conjunctions = [
        _filters_to_conjunction_expr(filter_set, schema) for filter_set in filters
    ]
    not_none_conjunctions = [c for c in conjunctions if c is not None]
    if not not_none_conjunctions:
        return None

    return pl_any(not_none_conjunctions)


def _filter_to_expr(filter: Filter, schema: pl.Schema | None = None) -> pl.Expr:
    """Convert a single filter to a polars expression.

    Args:
        filter (Filter): A filter object containing the column, operator, and value.
        schema (pl.Schema | None, optional): schema of the table, used to coerce
            string filter values for temporal columns. Defaults to None.

    Raises:
        ValueError: If the operator is not supported.

    Returns:
        pl.Expr: A polars expression that represents the filter condition.
    """
    dtype = schema.get(filter.column) if schema is not None else None
    if filter.operator == "=":
        return pl.col(filter.column) == _coerce_temporal_value(filter.value, dtype)
    elif filter.operator == "!=":
        return pl.col(filter.column) != _coerce_temporal_value(filter.value, dtype)
    elif filter.operator == "<":
        return pl.col(filter.column) < _coerce_temporal_value(filter.value, dtype)
    elif filter.operator == ">":
        return pl.col(filter.column) > _coerce_temporal_value(filter.value, dtype)
    elif filter.operator == "<=":
        return pl.col(filter.column) <= _coerce_temporal_value(filter.value, dtype)
    elif filter.operator == ">=":
        return pl.col(filter.column) >= _coerce_temporal_value(filter.value, dtype)
    elif filter.operator == "in":
        # ponytail: only coerce "in"/"not in"; "includes"/"includes any"/"includes all"
        # (list-membership filters) aren't covered since no failure was observed there.
        return pl.col(filter.column).is_in(
            [_coerce_temporal_value(v, dtype) for v in filter.value]
        )
    elif filter.operator == "not in":
        return ~pl.col(filter.column).is_in(
            [_coerce_temporal_value(v, dtype) for v in filter.value]
        )
    elif filter.operator == "is null":
        return pl.col(filter.column).is_null()
    elif filter.operator == "is not null":
        return pl.col(filter.column).is_not_null()
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
        schema: dict[str, pl.DataType] | None = None,
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
            schema (dict[str, pl.DataType] | None, optional): explicit schema for the table.
                When provided, allows scan_parquet to succeed even when the S3 path
                contains no files (returns an empty DataFrame with this schema).
                Defaults to None.

        Raises:
            ValueError: if the partitioning_scheme is not a valid PartitioningScheme.
        """
        if not isinstance(partitioning_scheme, PartitioningScheme):
            raise ValueError(f"Invalid partitioning scheme, got {partitioning_scheme}")

        self.name = name
        self.uri = uri
        self.partitioning = partitioning
        self.partitioning_scheme = partitioning_scheme
        self.schema = schema

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
                operator=filter.operator,
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
                    name=key,
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

        hive_schema = (
            {partition.column: partition.col_type for partition in remaining_partitions}
            if remaining_partitions
            else None
        )

        df = pl.scan_parquet(
            uri,
            hive_partitioning=len(remaining_partitions) > 0,
            allow_missing_columns=True,
            storage_options=storage_options,
            hive_schema=hive_schema,
            schema=self.schema,
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
            filter_expr = _filters_to_expr(remaining_filters, df.collect_schema())
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

            # An emptied branch is true within the selected partition. Keeping
            # the other OR branches would incorrectly restrict its rows, and
            # their predicates cannot justify pruning any deeper partitions.
            if any(not filter_set for filter_set in filters):
                filters = [[]]
                break

        uri = path.join(
            uri, ""
        )  # trailing slash prevents inclusion of partitions that are subsets of other partitions

        return (uri, partitions, filters, applied_filters)
