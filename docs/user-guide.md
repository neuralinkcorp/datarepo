# User Guide

## Core concepts

### Tables

A table in datarepo is a Python function that returns an `NlkDataFrame`. An `NlkDataFrame` is a thin wrapper of the [polars LazyFrame](https://docs.pola.rs/py-polars/html/reference/lazyframe/index.html). Tables are the fundamental building blocks for accessing and querying data. Tables can be backed by [DeltaLake](https://delta.io/) tables, [Parquet tables](https://parquet.apache.org/), or pure Python functions.

#### Delta Lake tables
```python
from datarepo.core import DeltalakeTable
import pyarrow as pa

# Define the schema
schema = pa.schema([
    ("p_partkey", pa.int64()),
    ("p_name", pa.string()),
    ("p_mfgr", pa.string()),
    ("p_brand", pa.string()),
    ("p_type", pa.string()),
    ("p_size", pa.int32()),
    ("p_container", pa.string()),
    ("p_retailprice", pa.decimal128(12, 2)),
    ("p_comment", pa.string()),
])

# Create the table
part = DeltalakeTable(
    name="part",
    uri="s3://my-bucket/tpc-h/part",
    schema=schema,
    docs_filters=[
        Filter("p_partkey", "=", 1),
        Filter("p_brand", "=", "Brand#1"),
    ],
    unique_columns=["p_partkey"],
    description="""
    Part information from the TPC-H benchmark.
    Contains details about parts including name, manufacturer, brand, and retail price.
    """,
    table_metadata_args={
        "data_input": "Part catalog data from manufacturing systems, updated daily",
        "latency_info": "Daily batch updates from manufacturing ERP system",
        "example_notebook": "https://example.com/notebooks/part_analysis.ipynb",
    },
)
```

#### Schema versioning
When schema or business logic evolves, older data may remain at versioned URIs (for example `s3://my-bucket/tpc-h/part/v1`). Use `schema_versions` to bind each version label to a PyArrow schema and resolve reads to `{uri}/{version}`:

```python
v1_schema = pa.schema([
    ("p_partkey", pa.int64()),
    ("p_name", pa.string()),
    ("p_size", pa.int32()),
])

v2_schema = pa.schema([
    ("p_partkey", pa.int64()),
    ("p_name", pa.string()),
    ("p_size", pa.int32()),
    ("p_comment", pa.string()),
])

part = DeltalakeTable(
    name="part",
    uri="s3://my-bucket/tpc-h/part",
    schema=v2_schema,
    schema_versions={"v1": v1_schema, "v2": v2_schema},
    default_schema_version="v2",
)

# Latest schema (v2) from s3://my-bucket/tpc-h/part/v2
>>> df_v2 = part()

# Prior schema (v1) from s3://my-bucket/tpc-h/part/v1
>>> df_v1 = part.at_version("v1")()
>>> df_v1_inline = part(schema_version="v1")
```

Existing tables without `schema_versions` behave exactly as before.

#### Parquet tables
```python
from datarepo.core import ParquetTable, Partition, PartitioningScheme
import pyarrow as pa

# Create the table
partsupp = ParquetTable(
    name="partsupp",
    uri="s3://my-bucket/tpc-h/partsupp",
    partitioning=[
        Partition(column="ps_partkey", col_type=pl.Int64),
        Partition(column="ps_suppkey", col_type=pl.Int64),
    ],
    partitioning_scheme=PartitioningScheme.HIVE,
    docs_filters=[
        Filter("ps_partkey", "=", 1),
        Filter("ps_suppkey", "=", 1),
    ],
    description="""
    Part supplier information from the TPC-H benchmark.
    Contains details about parts supplied by suppliers including available quantity and supply cost.
    """,
    table_metadata_args={
        "data_input": "Supplier inventory and pricing data from procurement systems",
        "latency_info": "Real-time updates from supplier inventory management systems",
        "example_notebook": "https://example.com/notebooks/supplier_analysis.ipynb",
    },
)
```

#### Function tables
Function tables are created using the `@table` decorator and allow you to define custom data access logic:

```python
from datarepo.core import table
import polars as pl

@table(
    data_input="Supplier master data from vendor management system <code>/api/suppliers/master</code> endpoint",
    latency_info="Updated weekly by the supplier_master_sync DAG on Airflow",
)
def supplier() -> NlkDataFrame:
    """Supplier information from the TPC-H benchmark."""
    data = {
        "s_suppkey": [1, 2, 3, 4, 5],
        "s_name": ["Supplier#1", "Supplier#2", "Supplier#3", "Supplier#4", "Supplier#5"],
        "s_address": ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm St", "654 Maple Dr"],
        "s_nationkey": [1, 1, 2, 2, 3],
        "s_phone": ["555-0001", "555-0002", "555-0003", "555-0004", "555-0005"],
        "s_acctbal": [1000.00, 2000.00, 3000.00, 4000.00, 5000.00],
        "s_comment": ["Comment 1", "Comment 2", "Comment 3", "Comment 4", "Comment 5"]
    }
    return NlkDataFrame(data)
```

### Databases

A datarepo database is a Python module that contains tables. There are two main ways to create databases:

#### Module database
A module database wraps a Python module containing table definitions:

```python
# tpch_tables.py
from datarepo.core import table

@table
def supplier():
    """Supplier information."""
    return NlkDataFrame(...)

@table
def partsupp():
    """Part supplier relationship information."""
    return NlkDataFrame(...)

# Using the database
from datarepo.core import ModuleDatabase
import tpch_tables

db = ModuleDatabase(tpch_tables)

# Query data
>>> df = db.supplier()
>>> df.head()
shape: (5, 7)
┌──────────┬───────────┬────────────┬────────────┬──────────┬──────────┬──────────┐
│ s_suppkey│ s_name    │ s_address  │ s_nationkey│ s_phone  │ s_acctbal│ s_comment│
├──────────┼───────────┼────────────┼────────────┼──────────┼──────────┼──────────┤
│ 1        │ Supplier#1│ 123 Main St│ 1          │ 555-0001 │ 1000.00  │ Comment 1│
│ 2        │ Supplier#2│ 456 Oak Ave│ 1          │ 555-0002 │ 2000.00  │ Comment 2│
│ 3        │ Supplier#3│ 789 Pine Rd│ 2          │ 555-0003 │ 3000.00  │ Comment 3│
│ 4        │ Supplier#4│ 321 Elm St │ 2          │ 555-0004 │ 4000.00  │ Comment 4│
│ 5        │ Supplier#5│ 654 Maple Dr│ 3         │ 555-0005 │ 5000.00  │ Comment 5│
└──────────┴───────────┴────────────┴────────────┴──────────┴──────────┴──────────┘
```

### Catalogs

A catalog is a Python module that is a collection of databases.

```python
from datarepo.core import Catalog, ModuleDatabase
import tpch_tables

# Create a catalog
dbs = {"tpc-h": ModuleDatabase(tpch_tables)}
TPCHCatalog = Catalog(dbs)

# Query data across databases
>>> supplier = TPCHCatalog.db("tpc-h").supplier()
>>> partsupp = TPCHCatalog.db("tpc-h").partsupp()

# Join data across databases
>>> joined = supplier.join(partsupp, left_on="s_suppkey", right_on="ps_suppkey")
>>> joined.head()
shape: (5, 12)
┌──────────┬───────────┬────────────┬────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬────────────┬─────────────┬────────── ┐
│ s_suppkey│ s_name    │ s_address  │ s_nationkey│ s_phone  │ s_acctbal│ s_comment│ps_partkey│ps_suppkey│ps_availqty │ps_supplycost│ps_comment │
├──────────┼───────────┼────────────┼────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────────┼─────────────┼───────────┤
│ 1        │ Supplier#1│ 123 Main St│ 1          │ 555-0001 │ 1000.00  │ Comment 1│ 1        │ 1        │ 100        │ 100.00      │ Part 1    │
│ 2        │ Supplier#2│ 456 Oak Ave│ 1          │ 555-0002 │ 2000.00  │ Comment 2│ 2        │ 2        │ 200        │ 200.00      │ Part 2    │
│ 3        │ Supplier#3│ 789 Pine Rd│ 2          │ 555-0003 │ 3000.00  │ Comment 3│ 3        │ 3        │ 300        │ 300.00      │ Part 3    │
│ 4        │ Supplier#4│ 321 Elm St │ 2          │ 555-0004 │ 4000.00  │ Comment 4│ 4        │ 4        │ 400        │ 400.00      │ Part 4    │
│ 5        │ Supplier#5│ 654 Maple D│  3         │ 555-0005 │ 5000.00  │ Comment 5│ 5        │ 5        │ 500        │ 500.00      │ Part 5    │
└──────────┴───────────┴────────────┴────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────────┴─────────────┴───────────┘
```

## Querying data

datarepo provides a consistent interface for querying data across all table types:

```python
# Filter data
>>> df = db.supplier(filters=[("s_nationkey", "=", 1)])

# Select columns
>>> df = db.supplier(columns=["s_suppkey", "s_name"])

# Complex queries
>>> df = db.supplier(
...     filters=[
...         ("s_nationkey", "=", 1),
...         ("s_acctbal", ">=", 1000.00),
...     ],
...     columns=["s_suppkey", "s_name", "s_acctbal"],
... )
```

## Advanced features

### Caching
DeltaLake tables support caching to improve performance:

```python
from datarepo.core.tables import DeltaCacheOptions

# Configure caching
cache_options = DeltaCacheOptions(
    file_cache_path="~/.datarepo/cache",
    file_cache_last_checkpoint_valid_duration="30m",
)

# Use caching
>>> df = db.supplier(cache_options=cache_options)
```

### Custom columns
You can add custom computed columns to tables:

```python
# Add a custom column
supplier = DeltalakeTable(
    name="supplier",
    uri="s3://my-bucket/tpc-h/supplier",
    schema=schema,
    extra_cols=[
        (pl.col("s_acctbal") * 1.1, "s_acctbal_with_tax"),
    ],
)

# Query with custom column
>>> df = supplier(columns=["s_suppkey", "s_acctbal_with_tax"])
```