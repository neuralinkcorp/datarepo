from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, cast

import logging
import polars as pl
import pyarrow as pa
import clickhouse_connect

from datarepo.core.dataframe import NlkDataFrame
from datarepo.core.tables.filters import Filter, InputFilters, normalize_filters
from datarepo.core.tables.metadata import (
    TableColumn,
    TableMetadata,
    TableProtocol,
    TableSchema,
)
from datarepo.core.tables.util import RoapiOptions
from datarepo.core.tables.util import format_value_for_sql


logger = logging.getLogger(__name__)


@dataclass
class ClickHouseTableConfig:
    """Configuration for connecting to ClickHouse."""

    host: str
    port: int = 8123
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "default"
    table_name: Optional[str] = None
    secure: bool = False
    verify: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)

    def get_uri(self) -> str:
        """Construct the URI for the ClickHouse table.

        Returns:
            str: URI for the ClickHouse table.
        """
        # check if username and password are provided
        if not self.username or not self.password:
            return f"clickhouse://{self.host}:{self.port}/{self.database}"
        return f"clickhouse://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_client(self) -> clickhouse_connect.driver.Client:
        """Get a ClickHouse client for this configuration.

        Returns:
            clickhouse_connect.driver.Client: ClickHouse client instance.
        """
        return clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            username=self.username or "default",
            password=self.password or "",
            database=self.database,
            secure=self.secure,
            verify=self.verify,
            settings=self.settings,
        )


def make_clickhouse_config(
    table_name: str | None = None,
    host: str | None = None,
    port: int | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> ClickHouseTableConfig:
    """Create a ClickHouse config from environment variables or explicit values.

    Args:
        table_name: The actual table name in ClickHouse. If not provided,
            queries will use the ClickHouseTable.name as the table name.

    Environment variables (used if explicit values not provided):
        CLICKHOUSE_HOST: ClickHouse server hostname (default: localhost)
        CLICKHOUSE_PORT: ClickHouse HTTP port (default: 8123). Use 8443 with
            secure=True for TLS.
        CLICKHOUSE_USER: ClickHouse username (default: default)
        CLICKHOUSE_PASSWORD: ClickHouse password
        CLICKHOUSE_DATABASE: ClickHouse database (default: default)
    """
    return ClickHouseTableConfig(
        host=host or os.environ.get("CLICKHOUSE_HOST", "localhost"),
        port=port or int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        username=username or os.environ.get("CLICKHOUSE_USER", "default"),
        password=password or os.environ.get("CLICKHOUSE_PASSWORD", ""),
        database=database or os.environ.get("CLICKHOUSE_DATABASE", "default"),
        table_name=table_name,
    )


class ClickHouseTable(TableProtocol):
    """A table implementation that reads data from ClickHouse."""

    def __init__(
        self,
        name: str,
        schema: pa.Schema,
        description: str = "",
        config: ClickHouseTableConfig | None = None,
        docs_filters: List[Filter] | None = None,
        docs_columns: Optional[List[str]] = None,
        roapi_opts: RoapiOptions | None = None,
        unique_columns: Optional[List[str]] = None,
        table_metadata_args: Optional[Dict[str, Any]] = None,
        stats_cols: Optional[List[str]] = None,
    ):
        """Initialize the ClickHouseTable.

        Example usage:
            ```python
            from datarepo.core.tables import ClickHouseTable, ClickHouseTableConfig

            config = ClickHouseTableConfig(
                host="localhost",
                port=8443,
                username="user",
                password="password",
                database="default",
                table_name="delta_my_table",  # actual table in ClickHouse
                secure=True,
                verify=True,
                settings={"max_result_rows": 1000}
            )

            table = ClickHouseTable(
                name="my_table",  # logical/API name
                schema=pa.schema(
                    [
                        ("implant_id", pa.int64()),
                        ("date", pa.string()),
                        ("uniq", pa.string()),
                        ("value", pa.int64()),
                    ]
                ),
                config=config,
                description="My ClickHouse table",
                docs_filters=[...],
                docs_columns=[...],
                unique_columns=["uniq"],
                table_metadata_args={"answer": "42"},
                stats_cols=["implant_id"]
            )
            ```

        Args:
            name: Logical name of the table (used in Python API/code)
            schema: Schema of the table
            config: Configuration for connecting to ClickHouse. The config.table_name
                specifies the actual ClickHouse table name (useful when migrating
                from delta-backed to native tables). Falls back to `name` if not set.
            description: Description of the table for documentation
            docs_filters: Filters to show in documentation
            docs_columns: Columns to show in documentation
            roapi_opts: Options for ROAPI integration
            unique_columns: Columns to use for deduplication
            table_metadata_args: Additional metadata arguments
            stats_cols: Statistics columns, used to define columns that have statistics
        """
        self.name = name
        self.schema = schema
        self.config = config or make_clickhouse_config()
        self.unique_columns = unique_columns or []
        self.docs_filters = docs_filters or []
        self.docs_columns = docs_columns
        self.stats_cols = stats_cols or []
        # for roapi
        self.uri = self.config.get_uri()

        self.table_metadata = TableMetadata(
            table_type="CLICKHOUSE",
            description=description,
            docs_args={"filters": self.docs_filters, "columns": self.docs_columns},
            roapi_opts=roapi_opts or RoapiOptions(),
            **(table_metadata_args or {}),
        )

    def get_schema(self) -> TableSchema:
        """Generate and return the schema of the table, including columns.

        Returns:
            TableSchema: table schema containing column information.
        """
        schema = self.schema

        columns = [
            TableColumn(
                name=name,
                type=str(schema.field(name).type),
                readonly=False,
                filter_only=False,
                has_stats=name in self.stats_cols,
            )
            for name in schema.names
        ]

        return TableSchema(
            partitions=[],  # Clickhouse does not have partitions exposed in schema.
            columns=columns,
        )

    def _build_query(
        self,
        filters: InputFilters | None = None,
        columns: Optional[List[str]] = None,
        config: ClickHouseTableConfig | None = None,
    ) -> str:
        """Build a SQL query for the ClickHouse table.

        Args:
            filters (InputFilters, optional): Filters to apply to the query. Defaults to None.
            columns (Optional[List[str]], optional): Columns to select in the query. Defaults to None.

        Returns:
            str: SQL query string to select data from the ClickHouse table.

        Raises:
            ValueError: If a filter uses an unsupported operator. Unary null
                predicates ignore Filter.value.
        """
        config = config or self.config
        table_name = config.table_name or self.name

        column_expr = "*"
        if columns:
            valid_columns = [c for c in columns if c in self.schema.names]
            if valid_columns:
                column_expr = ", ".join(f"`{c}`" for c in valid_columns)
            else:
                logger.warning(
                    f"No valid columns provided for table {self.name}. Using '*' to select all columns."
                )

        # Build WHERE clause from filters if provided
        where_clause = ""
        if filters:
            normalized_filters = normalize_filters(filters)
            filter_expressions = []

            for filter_set in normalized_filters:
                set_expressions = []
                for f in filter_set:
                    if f.operator == "=":
                        set_expressions.append(
                            f"`{f.column}` = {format_value_for_sql(f.value)}"
                        )
                    elif f.operator == "!=":
                        set_expressions.append(
                            f"`{f.column}` != {format_value_for_sql(f.value)}"
                        )
                    elif f.operator == ">":
                        set_expressions.append(
                            f"`{f.column}` > {format_value_for_sql(f.value)}"
                        )
                    elif f.operator == "<":
                        set_expressions.append(
                            f"`{f.column}` < {format_value_for_sql(f.value)}"
                        )
                    elif f.operator == ">=":
                        set_expressions.append(
                            f"`{f.column}` >= {format_value_for_sql(f.value)}"
                        )
                    elif f.operator == "<=":
                        set_expressions.append(
                            f"`{f.column}` <= {format_value_for_sql(f.value)}"
                        )
                    elif f.operator == "in":
                        values = ", ".join(
                            format_value_for_sql(v) for v in cast(list, f.value)
                        )
                        set_expressions.append(f"`{f.column}` IN ({values})")
                    elif f.operator == "not in":
                        values = ", ".join(
                            format_value_for_sql(v) for v in cast(list, f.value)
                        )
                        set_expressions.append(f"`{f.column}` NOT IN ({values})")
                    elif f.operator == "is null":
                        set_expressions.append(f"`{f.column}` IS NULL")
                    elif f.operator == "is not null":
                        set_expressions.append(f"`{f.column}` IS NOT NULL")
                    elif f.operator in [
                        "contains",
                        "includes",
                        "includes any",
                        "includes all",
                    ]:
                        set_expressions.append(
                            f"`{f.column}` LIKE {format_value_for_sql(f.value)}"
                        )
                    else:
                        raise ValueError(f"Unsupported filter operator: {f.operator!r}")

                if set_expressions:
                    filter_expressions.append("(" + " AND ".join(set_expressions) + ")")

            if filter_expressions:
                where_clause = "WHERE " + " OR ".join(filter_expressions)

        return f"SELECT {column_expr} FROM `{config.database}`.`{table_name}` {where_clause}"

    def __call__(  # type: ignore[override]
        self,
        filters: InputFilters | None = None,
        columns: List[str] | None = None,
        config: ClickHouseTableConfig | None = None,
        **kwargs: Dict[str, Any],
    ) -> NlkDataFrame:
        """Query data from the ClickHouse table.

        Example usage:
            ``` py
            from datarepo.core.tables import ClickHouseTable, ClickHouseTableConfig
            config = ClickHouseTableConfig(...)
            table = ClickHouseTable(...)
            df = table(filters=[Filter("implant_id", "=", 123)], columns=["date", "value"])
            ```

        Args:
            filters: Filters to apply to the data.
            columns: Columns to select from the table.
            config: Optional configuration for the ClickHouse table.
            **kwargs: Additional arguments.

        Returns:
            NlkDataFrame: A lazy Polars DataFrame with the requested data.

        Raises:
            ValueError: If a filter uses an unsupported operator, before creating
                a backend client or executing a query.
        """
        config = config or self.config

        query = self._build_query(filters, columns, config)

        client = config.get_client()
        arrow_result = client.query_arrow(query)
        df = pl.from_arrow(arrow_result)
        if isinstance(df, pl.Series):
            df = df.to_frame()

        if self.unique_columns:
            available_unique_cols = [c for c in self.unique_columns if c in df.columns]
            if available_unique_cols:
                df = df.filter(pl.struct(available_unique_cols).is_last_distinct())

        return df.lazy()
