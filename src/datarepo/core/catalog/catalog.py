from types import ModuleType
from typing import Any, Protocol, cast
import warnings

from datarepo.core.dataframe import NlkDataFrame
from datarepo.core.tables.metadata import TableProtocol


class Database(Protocol):
    """A protocol for a database that provides access to tables."""

    def get_tables(self, show_deprecated: bool = False) -> dict[str, TableProtocol]:
        """Get a dictionary of tables in the database.

        Args:
            show_deprecated (bool, optional): Whether to include deprecated tables. Defaults to False.
        """
        ...

    def tables(self, show_deprecated: bool = False) -> list[str]:
        """Get a list of table names in the database.

        Args:
            show_deprecated (bool, optional): Whether to include deprecated tables. Defaults to False.

        Returns:
            list[str]: A list of table names.
        """
        return list(self.get_tables(show_deprecated).keys())

    def table(self, name: str, *args: Any, **kwargs: Any) -> NlkDataFrame:
        """Get a table from the database.

        Args:
            name (str): The name of the table.

        Returns:
            NlkDataFrame: The requested table.
        """
        ...


class ModuleDatabase(Database):
    """A database that is implemented as a Python module."""

    def __init__(self, db: ModuleType) -> None:
        """Initialize the ModuleDatabase.

        Example usage:
            ``` py
            import my_database_module
            db = ModuleDatabase(my_database_module)
            ```

        Args:
            db (ModuleType): The database module.
        """
        self.db = db

    def __getattr__(self, name: str):
        # HACK: to maintain backwards compatibility when accessing module attributes
        return self._get_table(name)

    def get_tables(self, show_deprecated: bool = False) -> dict[str, TableProtocol]:
        """Get a dictionary of tables in the database.

        Example usage:
            ``` py
            db = ModuleDatabase(my_database_module)
            tables = db.get_tables(show_deprecated=True)
            ```

        Args:
            show_deprecated (bool, optional): Whether to include deprecated tables. Defaults to False.

        Returns:
            dict[str, TableProtocol]: A dictionary of table names and their corresponding TableProtocol objects.
        """
        methods = dir(self.db)

        tables = {}
        for name in methods:
            table = self._get_table(name)
            if table is None:
                continue

            if table.table_metadata.is_deprecated and not show_deprecated:
                continue

            tables[name] = table

        return tables

    def table(self, name: str, *args: Any, **kwargs: Any) -> NlkDataFrame:
        """Get a table from the database.

        Example usage:
            ``` py
            db = ModuleDatabase(my_database_module)
            table = db.table("my_table")
            ```

        Args:
            name (str): The name of the table.

        Raises:
            KeyError: If the table is not found.

        Returns:
            NlkDataFrame: The requested table.
        """
        tbl = self._get_table(name)
        if tbl is None:
            raise KeyError(f"Table '{name}' not found in database")

        if tbl.table_metadata.is_deprecated:
            warnings.warn(f"The table '{name}' is deprecated", DeprecationWarning)

        return tbl(*args, **kwargs)

    def _get_table(self, name: str) -> TableProtocol | None:
        """Get a table from the database.

        Args:
            name (str): The name of the table.

        Returns:
            TableProtocol | None: The requested table or None if not found.
        """
        table = getattr(self.db, name)
        if not hasattr(table, "table_metadata"):
            return None

        return cast(TableProtocol, table)


class DatabaseWithGlobalArgs(Database):
    """A wrapper for a database that applies global arguments to all table calls."""

    def __init__(self, db: Database, global_args: dict[str, Any]):
        """Initialize the DatabaseWithGlobalArgs.

        Args:
            db (Database): The database to wrap.
            global_args (dict[str, Any]): The global arguments to apply to all table calls.
        """
        self.db = db
        self._global_args = global_args

    def get_tables(self, show_deprecated: bool = False) -> dict[str, TableProtocol]:
        """Get a dictionary of tables in the database.

        Args:
            show_deprecated (bool, optional): Whether to include deprecated tables. Defaults to False.

        Returns:
            dict[str, TableProtocol]: A dictionary of table names and their corresponding TableProtocol objects.
        """
        return self.db.get_tables(show_deprecated)

    def tables(self, show_deprecated: bool = False) -> list[str]:
        """Get a list of table names in the database.

        Args:
            show_deprecated (bool, optional): Whether to include deprecated tables. Defaults to False.

        Returns:
            list[str]: A list of table names.
        """
        return self.db.tables(show_deprecated)

    def table(self, name: str, *args: Any, **kwargs: Any) -> NlkDataFrame:
        """Get a table from the database.

        Args:
            name (str): The name of the table.

        Returns:
            NlkDataFrame: The requested table.
        """
        new_kwargs = {**self._global_args, **kwargs}
        return self.db.table(name, *args, **new_kwargs)


class Catalog:
    """A catalog that manages multiple databases and provides access to their tables."""

    def __init__(self, dbs: dict[str, Database]):
        """Initialize the Catalog.

        Args:
            dbs (dict[str, Database]): A dictionary of database names and their corresponding Database objects.
        """
        self._dbs = dbs
        self._global_args: dict[str, Any] | None = None

    def set_global_args(self, global_args: dict[str, Any]) -> None:
        """Set global arguments for all database queries.

        Args:
            global_args (dict[str, Any]): A dictionary of global arguments to apply to all database queries.
        """
        self._global_args = global_args

    def db(self, db_name: str) -> Database:
        """Get a database from the catalog.

        Args:
            db_name (str): The name of the database.

        Raises:
            KeyError: If the database is not found.

        Returns:
            Database: The requested database.
        """
        db = self._dbs.get(db_name)

        if db is None:
            raise KeyError(
                f"Database '{db_name}' not found. Available databases: {self.dbs()}"
            )

        if self._global_args is None:
            return db

        return DatabaseWithGlobalArgs(db, self._global_args)

    def dbs(self) -> list[str]:
        """Get a list of database names in the catalog.

        Returns:
            list[str]: A list of database names.
        """
        return list(self._dbs.keys())
