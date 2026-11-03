import datetime
from unittest.mock import MagicMock, patch

from botocore.exceptions import ProfileNotFound
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
    get_storage_options,
)

test_schema = pa.schema(
    [
        ("str_col", pa.string()),
        ("int_col", pa.int64()),
        ("list_col", pa.list_(pa.int64())),
        ("list_str_col", pa.list_(pa.string())),
        ("large_str_col", pa.large_string()),
        ("str_view_col", pa.string_view()),
        ("date_col", pa.date32()),
        ("ts_col", pa.timestamp("us")),
        ("ts_utc_col", pa.timestamp("us", tz="UTC")),
        ("list_date_col", pa.list_(pa.date32())),
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
                "(str_col like '%x''%' escape '\\')",
            ),
            # contains is a literal substring match: LIKE wildcards and the escape
            # character in the value are escaped
            (
                test_schema,
                Filter("str_col", "contains", "50%_x"),
                "(str_col like '%50\\%\\_x%' escape '\\')",
            ),
            (
                test_schema,
                Filter("str_col", "contains", "a\\b"),
                "(str_col like '%a\\\\b%' escape '\\')",
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
            # Other string types are quoted and escaped like pa.string()
            (
                test_schema,
                Filter("large_str_col", "=", "it's"),
                "(large_str_col = 'it''s')",
            ),
            (
                test_schema,
                Filter("str_view_col", "in", ("a", "b")),
                "(str_view_col in ('a', 'b'))",
            ),
            # Date columns get a DATE literal, from a string or a date
            (
                test_schema,
                Filter("date_col", "=", "2024-01-01"),
                "(date_col = DATE '2024-01-01')",
            ),
            (
                test_schema,
                Filter("date_col", "in", (datetime.date(2024, 1, 1),)),
                "(date_col in (DATE '2024-01-01'))",
            ),
            (
                test_schema,
                Filter("list_date_col", "includes", "2024-01-01"),
                "(array_contains(list_date_col, DATE '2024-01-01'))",
            ),
            # Timestamp columns are quoted, keeping any UTC offset of the value
            (
                test_schema,
                Filter("ts_col", ">=", datetime.datetime(2024, 1, 1, 13)),
                "(ts_col >= '2024-01-01 13:00:00')",
            ),
            (
                test_schema,
                Filter(
                    "ts_utc_col",
                    "<",
                    datetime.datetime(2024, 1, 1, 13, tzinfo=datetime.timezone.utc),
                ),
                "(ts_utc_col < '2024-01-01 13:00:00+00:00')",
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

    @patch("datarepo.core.tables.util.datarepo_config")
    @patch("datarepo.core.tables.util.boto3.Session")
    def test_get_storage_options_skips_missing_profile(
        self, mock_session: MagicMock, mock_config: MagicMock
    ):
        """A missing AWS profile is ignored instead of aborting credential lookup."""
        mock_config.DEFAULT_AWS_PROFILE = "missing-profile"

        default_session = MagicMock()
        default_session.get_credentials.return_value = None
        mock_session.side_effect = [
            default_session,
            ProfileNotFound(profile="missing-profile"),
        ]

        options = get_storage_options()

        assert options == {}
        assert mock_session.call_count == 2
        mock_session.assert_called_with(profile_name="missing-profile")
