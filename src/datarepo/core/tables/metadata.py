from dataclasses import dataclass
from typing import Any, Dict, Protocol, TypedDict

from datarepo.core.dataframe.frame import NlkDataFrame
from datarepo.core.tables.util import RoapiOptions


@dataclass
class TableMetadata:
    """
    Information about a table used for documentation generation.
    """

    table_type: str
    description: str
    docs_args: Dict[str, Any]
    latency_info: str | None = None
    example_notebook: str | None = None
    data_input: str | None = None
    is_deprecated: bool = False
    roapi_opts: RoapiOptions | None = None


class TablePartition(TypedDict):
    column_name: str
    type_annotation: str
    value: Any


class TableColumn(TypedDict):
    column: str
    type: str
    readonly: bool
    filter_only: bool
    has_stats: bool


@dataclass
class TableSchema:
    """TableSchema represents the schema of a table, including partitions and columns."""

    partitions: list[TablePartition]
    columns: list[TableColumn]


class TableProtocol(Protocol):
    # Properties used to generate the web catalog & roapi config
    table_metadata: TableMetadata

    def __call__(self, **kwargs: Dict[str, Any]) -> NlkDataFrame: ...

    def get_schema(self) -> TableSchema:
        """
        Returns the schema of the table, used to generate the web catalog.
        """
        ...
