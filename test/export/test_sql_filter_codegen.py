"""Exercises the actual compiled TypeScript SQL-filter code generator
(``formatSqlPredicate`` / ``genTableCode`` with ``formatSqlFilter=True``) in
``src/datarepo/export/static_site/src/lib/codegen.ts``.

Regression coverage for https://github.com/neuralinkcorp/datarepo/issues/77:
the SQL-string filter snippet did not escape quotes, did not escape LIKE
wildcards in `contains` values, rendered `in`/`not in` list values via
JavaScript's `Array.toString()`, and quoted strings only when
`type_annotation` was exactly `"str"`/`"string"` rather than by the value's
actual type.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SITE_DIR = Path(__file__).parents[2] / "src" / "datarepo" / "export" / "static_site"
LIB_DIR = SITE_DIR / "src" / "lib"


def _tsc_binary() -> str | None:
    local = SITE_DIR / "node_modules" / ".bin" / "tsc"
    if local.is_file():
        return str(local)
    return shutil.which("tsc")


requires_toolchain = pytest.mark.skipif(
    shutil.which("node") is None or _tsc_binary() is None,
    reason="Node.js and a tsc binary are required to compile and execute codegen.ts",
)


@pytest.fixture(scope="module")
def generate_sql_filter(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("codegen_sql")
    subprocess.run(
        [
            _tsc_binary(),
            "--target",
            "ES2019",
            "--module",
            "commonjs",
            "--skipLibCheck",
            "--outDir",
            str(out_dir),
            str(LIB_DIR / "codegen.ts"),
            str(LIB_DIR / "types.ts"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    runner = out_dir / "run.js"
    runner.write_text("""
        const { genTableCode } = require('./codegen.js');
        const partitions = JSON.parse(process.argv[2]);
        const code = genTableCode({
          catalog: { name: 'cat', package_name: null, metadata: null, databases: [] },
          database: { name: 'db', tables: [] },
          table: {
            name: 'tbl', description: '', partitions, columns: null,
            selected_columns: null, supports_sql_filter: true,
            table_type: 'PARQUET', latency_info: null, example_notebook: null,
            data_input: null,
          },
          formatSqlFilter: true,
        });
        process.stdout.write(code);
        """)

    def generate(partitions: list[dict]) -> str:
        result = subprocess.run(
            ["node", str(runner), json.dumps(partitions)],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    return generate


def _extract_filters_arg(code: str) -> str:
    """Extract and Python-eval the `filters="..."` argument's own string contents,
    so assertions check the predicate text, not the surrounding Python/escaping.

    Scans to the matching unescaped closing quote rather than splitting on the next
    comma, since a rendered `in`/`not in` predicate legitimately contains commas
    inside its own parentheses (e.g. `"id in (1, 2, 3)"`)."""
    marker = 'filters="'
    start = code.index(marker) + len(marker)
    i = start
    while code[i] != '"':
        if code[i] == "\\":
            i += 1
        i += 1
    # The generated code is otherwise well-formed Python; reuse Python's own
    # literal evaluator so this test does not reimplement string unescaping.
    return eval(code[start - 1 : i + 1])  # noqa: S307 - trusted, locally generated code


@requires_toolchain
class TestSqlFilterCodegen:
    def test_embedded_single_quote_is_escaped(self, generate_sql_filter):
        code = generate_sql_filter(
            [
                {
                    "column_name": "name",
                    "operator": "=",
                    "type_annotation": "str",
                    "value": "O'Brien",
                }
            ]
        )
        assert _extract_filters_arg(code) == "name = 'O''Brien'"

    def test_embedded_double_quote_does_not_break_python_string(
        self, generate_sql_filter
    ):
        code = generate_sql_filter(
            [
                {
                    "column_name": "name",
                    "operator": "=",
                    "type_annotation": "str",
                    "value": 'say "hi"',
                }
            ]
        )
        # The whole filters="..." argument must itself be a single, valid Python string literal.
        assert _extract_filters_arg(code) == "name = 'say \"hi\"'"

    def test_contains_escapes_like_wildcards(self, generate_sql_filter):
        code = generate_sql_filter(
            [
                {
                    "column_name": "name",
                    "operator": "contains",
                    "type_annotation": "str",
                    "value": "50%_x",
                }
            ]
        )
        assert _extract_filters_arg(code) == "name like '%50\\%\\_x%' escape '\\'"

    def test_in_renders_list_as_sql_tuple(self, generate_sql_filter):
        code = generate_sql_filter(
            [
                {
                    "column_name": "id",
                    "operator": "in",
                    "type_annotation": "int",
                    "value": [1, 2, 3],
                }
            ]
        )
        assert _extract_filters_arg(code) == "id in (1, 2, 3)"

    def test_not_in_with_strings_quotes_each_element(self, generate_sql_filter):
        code = generate_sql_filter(
            [
                {
                    "column_name": "name",
                    "operator": "not in",
                    "type_annotation": "str",
                    "value": ["a", "b'c"],
                }
            ]
        )
        assert _extract_filters_arg(code) == "name not in ('a', 'b''c')"

    def test_quoting_follows_runtime_type_not_just_type_annotation(
        self, generate_sql_filter
    ):
        # type_annotation is "large_string", not exactly "str"/"string", but the value is
        # still a JS string at runtime and must still be quoted.
        code = generate_sql_filter(
            [
                {
                    "column_name": "name",
                    "operator": "=",
                    "type_annotation": "large_string",
                    "value": "abc",
                }
            ]
        )
        assert _extract_filters_arg(code) == "name = 'abc'"

    def test_is_null_unaffected(self, generate_sql_filter):
        code = generate_sql_filter(
            [
                {
                    "column_name": "name",
                    "operator": "is null",
                    "type_annotation": "str",
                    "value": None,
                }
            ]
        )
        assert _extract_filters_arg(code) == "name is null"
