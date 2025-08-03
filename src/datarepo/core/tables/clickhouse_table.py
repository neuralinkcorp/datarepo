from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, cast

import logging
import polars as pl
import pyarrow as pa

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
    port: int = 8443
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "default"
    secure: bool = True
    verify: bool = True
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


class ClickHouseTable(TableProtocol):
    """A table implementation that reads data from ClickHouse."""

    def __init__(
        self,
        name: str,
        schema: pa.Schema,
        config: ClickHouseTableConfig,
        description: str = "",
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
                secure=True,
                verify=True,
                settings={"max_result_rows": 1000}
            )

            table = ClickHouseTable(
                name="my_table",
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
            name: Name of the table in ClickHouse
            schema: Schema of the table
            config: Configuration for connecting to ClickHouse
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
        self.config = config
        self.unique_columns = unique_columns or []
        self.docs_filters = docs_filters or []
        self.docs_columns = docs_columns
        self.stats_cols = stats_cols or []
        self.uri = config.get_uri()

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
                column=name,
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
    ) -> str:
        """Build a SQL query for the ClickHouse table.

        Args:
            filters (InputFilters, optional): Filters to apply to the query. Defaults to None.
            columns (Optional[List[str]], optional): Columns to select in the query. Defaults to None.

        Returns:
            str: SQL query string to select data from the ClickHouse table.
        """
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
                    elif f.operator in [
                        "contains",
                        "includes",
                        "includes any",
                        "includes all",
                    ]:
                        set_expressions.append(
                            f"`{f.column}` LIKE {format_value_for_sql(f.value)}"
                        )

                if set_expressions:
                    filter_expressions.append("(" + " AND ".join(set_expressions) + ")")

            if filter_expressions:
                where_clause = "WHERE " + " OR ".join(filter_expressions)

        return f"SELECT {column_expr} FROM `{self.config.database}`.`{self.name}` {where_clause}"

    def __call__(  # type: ignore[override]
        self,
        filters: InputFilters | None = None,
        columns: List[str] | None = None,
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
            **kwargs: Additional arguments.

        Returns:
            NlkDataFrame: A lazy Polars DataFrame with the requested data.
        """
        query = self._build_query(filters, columns)

        # use polars read database_uri to read the result of the query
        df = pl.read_database_uri(
            query=query,
            uri=self.uri,
            engine="connectorx",
        )

        return df.lazy()
