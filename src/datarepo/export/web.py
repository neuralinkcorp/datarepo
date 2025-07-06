from datarepo.core.catalog.catalog import Catalog, Database
from datarepo.core.tables.deltalake_table import DeltalakeTable
from datarepo.core.tables.metadata import TableProtocol
import json
import tempfile
import subprocess
from pathlib import Path
import os
import sys
import logging
import shutil
import importlib.resources

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def export_table(name: str, table: TableProtocol):
    """Export a table to a dictionary format suitable for web catalog generation.

    Args:
        name (str): name of the table, used as the key in the catalog.
        table (TableProtocol): table to export.

    Returns:
        dict[str, Any]: A dictionary representing the table's metadata,
        including its name, description, partitions, columns, and other relevant information.
    """
    table_info = table.table_metadata

    schema = table.get_schema()
    partitions, columns = schema.partitions, schema.columns

    return {
        "name": name,
        "description": table_info.description,
        "partitions": partitions,
        "columns": columns,
        "selected_columns": table_info.docs_args.get("columns", None),
        "supports_sql_filter": isinstance(table, DeltalakeTable),
        "table_type": table_info.table_type,
        "latency_info": table_info.latency_info,
        "example_notebook": table_info.example_notebook,
        "data_input": table_info.data_input,
    }


def export_database(name: str, database: Database):
    """Export a database to a dictionary format suitable for web catalog generation.

    Args:
        name (str): name of the database, used as the key in the catalog.
        database (Database): Database to export.

    Returns:
        dict[str, Any]: A dictionary representing the database's metadata,
    """
    tables = []

    for key, table in sorted(database.get_tables().items()):
        tables.append(export_table(key, table))

    return {
        "name": name,
        "tables": tables,
    }


def export_catalog(name: str, catalog: Catalog):
    """Export a catalog to a dictionary format suitable for web catalog generation.

    Args:
        name (str): name of the catalog, used as the key in the export.
        catalog (Catalog): Catalog to export.

    Returns:
        dict[str, Any]: A dictionary representing the catalog's metadata,
        including its name and a list of databases.
    """
    return {
        "name": name,
        "databases": [export_database(key, catalog.db(key)) for key in catalog.dbs()],
    }


def export_datarepo(catalogs: list[tuple[str, Catalog]]) -> dict:
    """Export the datarepo catalogs to a dictionary format.

    Args:
        catalogs (list[tuple[str, Catalog]]): List of tuples containing catalog names and their corresponding Catalog objects.

    Returns:
        dict: A dictionary containing a list of exported catalogs, each represented as a dictionary.
    """
    return {
        "catalogs": [export_catalog(name, catalog) for name, catalog in catalogs],
    }


def export_and_generate_site(
    catalogs: list[tuple[str, Catalog]], output_dir: str = "./dist"
):
    """
    Export the catalog to a JSON file and copy the precompiled static site files.

    Args:
        catalogs: List of (name, catalog) tuples to export
        output_dir: Directory where the static site will be generated (default: "./dist")
    """
    # Export catalog to a JSON file
    catalog_data = export_datarepo(catalogs)

    output_path = Path(output_dir)
    # Remove output directory if it exists to ensure idempotency
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "data.json", "w") as f:
        json.dump(catalog_data, f)

    project_root = Path(__file__).parent
    precompiled_dir = project_root / "static_site" / "precompiled"

    logger.info(f"Copying precompiled directory {precompiled_dir} to {output_path}")
    if not precompiled_dir.exists():
        raise FileNotFoundError(
            f"Could not find precompiled directory. Make sure you're running from the project root or the package is properly installed."
        )

    shutil.copytree(precompiled_dir, output_path, dirs_exist_ok=True)
