"""Exercise the actual TypeScript generator with exported synthetic metadata."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import polars as pl
import pyarrow as pa
import pytest

from datarepo.core import Catalog, Filter, ModuleDatabase
from datarepo.core.tables.deltalake_table import DeltalakeTable
from datarepo.core.tables.parquet_table import ParquetTable
from datarepo.export import web


@pytest.fixture(scope="module")
def generate_code(tmp_path_factory):
    site = Path(web.__file__).parent / "static_site"
    tsc = os.environ.get("DATAREPO_TSC", str(site / "node_modules/.bin/tsc"))
    assert shutil.which("node"), "Node.js is required to test catalog code generation"
    assert Path(
        tsc
    ).is_file(), "Build/install the static site before running these tests"
    output = tmp_path_factory.mktemp("codegen")
    subprocess.run(
        [
            tsc,
            "--target",
            "ES2019",
            "--module",
            "commonjs",
            "--skipLibCheck",
            "--outDir",
            str(output),
            str(site / "src/lib/codegen.ts"),
            str(site / "src/lib/types.ts"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    def generate(catalog, sql=False, js_number=None):
        # Use the same JSON boundary as export_and_generate_site/data.json.
        exported = web.export_datarepo([("SampleCatalog", catalog)])["catalogs"][0]
        database = exported["databases"][0]
        payload = dict(
            catalog=exported,
            database=database,
            table=database["tables"][0],
            formatSqlFilter=sql,
        )
        result = subprocess.run(
            [
                "node",
                "-e",
                "const fs = require('node:fs');"
                "const {genTableCode} = require(process.argv[1]);"
                "const payload = JSON.parse(fs.readFileSync(0, 'utf8'));"
                "if (process.argv[2]) payload.table.partitions[0].value = Number(process.argv[2]);"
                "process.stdout.write(genTableCode(payload));",
                str(output / "codegen.js"),
                *([js_number] if js_number is not None else []),
            ],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    return generate


def make_catalog(table, database_name="db", table_name="rows"):
    module = ModuleType("synthetic_tables")
    setattr(module, table_name, table)
    return Catalog(
        {database_name: ModuleDatabase(module)}, package_name="synthetic_catalog"
    )


def execute_synthetic(code, catalog, monkeypatch):
    module = ModuleType("synthetic_catalog")
    module.SampleCatalog = catalog
    monkeypatch.setitem(sys.modules, "synthetic_catalog", module)
    namespace = {}
    # All metadata and names in these tests are authored synthetic fixtures.
    exec(code, namespace)
    return namespace["df"]


def assert_same_value(actual, expected):
    assert type(actual) is type(expected)
    assert actual == expected
    if isinstance(expected, float) and expected == 0:
        assert math.copysign(1, actual) == math.copysign(1, expected)
    if isinstance(expected, list):
        for left, right in zip(actual, expected):
            assert_same_value(left, right)


@pytest.mark.parametrize(
    "value,arrow_type,operator",
    [
        ("alpha", pa.string(), "="),
        (42, pa.int64(), ">="),
        (1.25, pa.float64(), "<"),
        (-0.0, pa.float64(), "="),
        (1e-9, pa.float64(), "="),
        (9007199254740991, pa.int64(), "="),
        pytest.param(1.0000000000000001e18, pa.float64(), "=", id="unsafe-float"),
        pytest.param(
            -1.0000000000000001e18, pa.float64(), "=", id="unsafe-negative-float"
        ),
        pytest.param(1e20, pa.float64(), "=", id="unsafe-integral-float"),
        pytest.param(
            [1.0000000000000001e18, -1.0000000000000001e18],
            pa.float64(),
            "in",
            id="unsafe-float-list",
        ),
        pytest.param(
            [[1.0000000000000001e18], [-0.0, 1.25]],
            pa.list_(pa.float64()),
            "in",
            id="unsafe-nested-float-list",
        ),
        pytest.param(
            [5e-324, 1.7976931348623157e308],
            pa.float64(),
            "in",
            id="floating-extremes",
        ),
        (True, pa.bool_(), "="),
        (False, pa.bool_(), "!="),
        (None, pa.string(), "="),
        ([], pa.int64(), "in"),
        ([1, 2], pa.int64(), "in"),
        (["a", "b"], pa.string(), "not in"),
        ([True, False, None], pa.bool_(), "in"),
        ([[1, 2], []], pa.list_(pa.int64()), "in"),
        ("ACME \"West\" and 'East'", pa.string(), "="),
        (r"C:\new\table", pa.string(), "="),
        ("line1\nline2\tend\r\b\f\x00", pa.string(), "="),
        ("café λ 🧠", pa.string(), "="),
    ],
)
def test_exported_filter_literal(
    generate_code, monkeypatch, value, arrow_type, operator
):
    table = DeltalakeTable(
        name="rows",
        uri="unused",
        schema=pa.schema([("value", arrow_type)]),
        docs_filters=[Filter("value", operator, value)],
    )
    code = generate_code(make_catalog(table))
    recording_catalog = Mock()
    execute_synthetic(code, recording_catalog, monkeypatch)
    recording_catalog.db.assert_called_once_with("db")
    args, kwargs = recording_catalog.db.return_value.table.call_args
    assert args[0] == "rows"
    assert len(args) == 2
    assert kwargs == {}
    assert len(args[1]) == 1
    observed = args[1][0]
    assert isinstance(observed, Filter)
    assert observed.column == "value"
    assert observed.operator == operator
    assert_same_value(observed.value, value)


@pytest.mark.parametrize("operator", ["is null", "is not null"])
def test_unary_filter_ignores_value(generate_code, monkeypatch, operator):
    table = DeltalakeTable(
        name="rows",
        uri="unused",
        schema=pa.schema([("value", pa.string())]),
        docs_filters=[Filter("value", operator, {"ignored": True})],
    )
    recording_catalog = Mock()
    execute_synthetic(
        generate_code(make_catalog(table)), recording_catalog, monkeypatch
    )
    args, _ = recording_catalog.db.return_value.table.call_args
    assert args[1] == (Filter("value", operator, None),)


@pytest.mark.parametrize(
    "value",
    [
        {"unexpected": 1},
        [{"unexpected": 1}],
        [1, [{"unexpected": 1}]],
        [[], {"unexpected": 1}, None],
    ],
)
def test_unsupported_filter_value_shows_placeholder(generate_code, value):
    table = DeltalakeTable(
        name="rows",
        uri="unused",
        schema=pa.schema(
            [("before", pa.string()), ("value", pa.string()), ("after", pa.string())]
        ),
        docs_filters=[
            Filter("before", "=", "supported"),
            Filter("value", "in", value),
            Filter("after", "!=", "other"),
        ],
    )
    code = generate_code(make_catalog(table))
    assert code == "# cannot render this filter value"
    namespace = {}
    exec(code, namespace)
    assert "df" not in namespace


@pytest.mark.parametrize("js_number", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_filter_value_shows_placeholder(generate_code, js_number):
    table = DeltalakeTable(
        name="rows",
        uri="unused",
        schema=pa.schema([("value", pa.float64())]),
        docs_filters=[Filter("value", "=", 0.0)],
    )
    assert generate_code(make_catalog(table), js_number=js_number) == (
        "# cannot render this filter value"
    )


def test_names_order_and_selected_columns(generate_code, monkeypatch):
    names = ['a"b', "line\nend", r"c\name"]
    table_name, db_name = 'table"\n', 'database"\\'
    filters = [Filter(names[0], "in", ['x"', "y"]), Filter(names[1], ">=", 2)]
    table = DeltalakeTable(
        name=table_name,
        uri="unused",
        schema=pa.schema(
            [(names[0], pa.string()), (names[1], pa.int64()), (names[2], pa.string())]
        ),
        docs_filters=filters,
        docs_columns=names[::-1],
    )
    code = generate_code(make_catalog(table, db_name, table_name))
    recording_catalog = Mock()
    execute_synthetic(code, recording_catalog, monkeypatch)
    recording_catalog.db.assert_called_once_with(db_name)
    recording_catalog.db.return_value.table.assert_called_once_with(
        table_name, tuple(filters), columns=names[::-1]
    )


@pytest.mark.parametrize("columns", [None, []])
def test_no_filters(generate_code, monkeypatch, columns):
    table = DeltalakeTable(
        name="rows",
        uri="unused",
        schema=pa.schema([("value", pa.int64())]),
        docs_columns=columns,
    )
    recording_catalog = Mock()
    execute_synthetic(
        generate_code(make_catalog(table)), recording_catalog, monkeypatch
    )
    expected_kwargs = {} if columns is None else {"columns": []}
    recording_catalog.db.return_value.table.assert_called_once_with(
        "rows", **expected_kwargs
    )


def test_sql_output_control(generate_code):
    table = DeltalakeTable(
        name="rows",
        uri="unused",
        schema=pa.schema([("name", pa.string()), ("size", pa.int64())]),
        docs_filters=[Filter("name", "contains", "Brand"), Filter("size", ">=", 10)],
    )
    assert generate_code(make_catalog(table), sql=True) == (
        "from synthetic_catalog import SampleCatalog\n"
        "from datarepo.core import Filter\n\n"
        'df = SampleCatalog.db("db").table(\n'
        '    "rows",\n'
        "    filters=\"name like '%Brand%' and size >= 10\",\n"
        ").collect()"
    )


@pytest.mark.parametrize(
    "values,expected_ids", [([r"C:\new", 'ACME "West"'], [1, 2]), ([], [])]
)
def test_generated_query_reads_local_parquet(
    generate_code, monkeypatch, tmp_path, values, expected_ids
):
    path = tmp_path / "synthetic.parquet"
    pl.DataFrame(
        {"id": [1, 2, 3], "name": [r"C:\new", 'ACME "West"', "other"]}
    ).write_parquet(path)
    monkeypatch.setattr(
        "datarepo.core.tables.parquet_table.get_storage_options", lambda **_kwargs: {}
    )
    table = ParquetTable(
        name="rows",
        uri=str(path),
        partitioning=[],
        docs_filters=[Filter("name", "in", values)],
        docs_columns=["id", "name"],
    )
    catalog = make_catalog(table)
    result = execute_synthetic(generate_code(catalog), catalog, monkeypatch)
    assert result.columns == ["id", "name"]
    assert result["id"].to_list() == expected_ids
    assert result["name"].to_list() == values
