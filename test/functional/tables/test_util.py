import pyarrow as pa
import pytest

from datarepo.core.tables.filters import (
    InputFilters,
    NormalizedFilters,
    normalize_filters,
)
from datarepo.core.tables.util import (
    Filter,
    filter_to_sql_expr,
    filters_to_sql_predicate,
)

test_schema = pa.schema(
    [
        ("str_col", pa.string()),
        ("int_col", pa.int64()),
        ("list_col", pa.list_(pa.int64())),
        ("list_str_col", pa.list_(pa.string())),
    ]
)


class TestUtil:
    @pytest.mark.parametrize(
        ("schema", "f", "expected"),
        [
            (test_schema, Filter("int_col", "=", 123), "(int_col = 123)"),
            (test_schema, Filter("int_col", "=", "123"), "(int_col = 123)"),
            # A tuple with a single element should not have a comma in SQL
            (test_schema, Filter("int_col", "in", (1,)), "(int_col in (1))"),
            (test_schema, Filter("int_col", "in", (1, 2)), "(int_col in (1, 2))"),
            (
                test_schema,
                Filter("int_col", "not in", (1, 2)),
                "(int_col not in (1, 2))",
            ),
            # String columns should be handled to add single quotes
            (test_schema, Filter("str_col", "=", "x"), "(str_col = 'x')"),
            (test_schema, Filter("str_col", "in", ("val1",)), "(str_col in ('val1'))"),
            (
                test_schema,
                Filter("str_col", "in", ("val1", "val2")),
                "(str_col in ('val1', 'val2'))",
            ),
            (
                test_schema,
                Filter("str_col", "contains", "x'"),
                "(str_col like '%x''%')",
            ),
            # Test list columns
            (
                test_schema,
                Filter("list_col", "includes", 1),
                "(array_contains(list_col, 1))",
            ),
            (
                test_schema,
                Filter("list_str_col", "includes", "x"),
                "(array_contains(list_str_col, 'x'))",
            ),
            (
                test_schema,
                Filter("list_col", "includes all", (1, 2, 3)),
                "(array_contains(list_col, 1) and array_contains(list_col, 2) and array_contains(list_col, 3))",
            ),
            (
                test_schema,
                Filter("list_col", "includes any", (1, 2, 3)),
                "(array_contains(list_col, 1) or array_contains(list_col, 2) or array_contains(list_col, 3))",
            ),
        ],
    )
    def test_filter_to_expr(self, schema: pa.Schema, f: Filter, expected: str):
        assert filter_to_sql_expr(schema, f) == expected

    def test_filter_to_expr_raises(self):
        with pytest.raises(ValueError) as e:
            filter_to_sql_expr(test_schema, Filter("invalid_col", "=", 0))

        assert "Invalid column name invalid_col" in str(e.value)

    @pytest.mark.parametrize(
        ("schema", "filters", "expected"),
        [
            (test_schema, [[Filter("str_col", "=", "x")]], "((str_col = 'x'))"),
            (
                test_schema,
                [[Filter("str_col", "=", "x"), Filter("int_col", "=", 123)]],
                "((str_col = 'x') and (int_col = 123))",
            ),
            (
                test_schema,
                [
                    [Filter("str_col", "=", "x")],
                    [Filter("int_col", "=", 123), Filter("int_col", "<", 456)],
                ],
                "((str_col = 'x')) or ((int_col = 123) and (int_col < 456))",
            ),
        ],
    )
    def test_filters_to_sql_predicate(
        self, schema: pa.Schema, filters: NormalizedFilters, expected: str
    ):
        assert filters_to_sql_predicate(schema, filters) == expected

    @pytest.mark.parametrize(
        ("filters", "expected"),
        [
            (None, []),
            ([], []),
            ([Filter("a", "=", 1)], [[Filter("a", "=", 1)]]),
            ((Filter("a", "=", 1),), [[Filter("a", "=", 1)]]),
            (
                [
                    [Filter("a", "=", 1)],
                    [Filter("b", "=", 2), Filter("c", "=", 3)],
                ],
                [
                    [Filter("a", "=", 1)],
                    [Filter("b", "=", 2), Filter("c", "=", 3)],
                ],
            ),
            (
                (
                    (Filter("a", "=", 1),),
                    (Filter("b", "=", 2), Filter("c", "=", 3)),
                ),
                [
                    [Filter("a", "=", 1)],
                    [Filter("b", "=", 2), Filter("c", "=", 3)],
                ],
            ),
        ],
    )
    def test_normalize_filters(
        self, filters: InputFilters, expected: NormalizedFilters
    ):
        assert normalize_filters(filters) == expected
