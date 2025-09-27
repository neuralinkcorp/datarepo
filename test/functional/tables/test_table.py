import pyarrow as pa

from datarepo.core.tables.util import (
    Filter,
    Partition,
    exactly_one_equality_filter,
)


class TestTable:
    def test_exactly_one_equality_filter(self):
        p = Partition(column="test", col_type=pa.int32())

        res = exactly_one_equality_filter(
            p,
            [
                Filter(column="test", operator="=", value=5956),
                Filter(column="test2", operator="<=", value="2024-04-06"),
            ],
        )

        assert res == ("test", "=", 5956)

    def test_exactly_one_equality_filter_none(self):
        p = Partition(column="test", col_type=pa.int32())

        res = exactly_one_equality_filter(
            p,
            [
                Filter(column="test2", operator="<=", value="2024-04-06"),
            ],
        )

        assert res is None

    def test_exactly_one_equality_filter_non_equality(self):
        p = Partition(column="test", col_type=pa.int32())

        res = exactly_one_equality_filter(
            p,
            [
                Filter(column="test", operator="<=", value=5956),
                ("test2", "<=", "2024-04-06"),  # type: ignore
            ],
        )

        assert res is None

    def test_exactly_one_equality_filter_multiple(self):
        p = Partition("test", pa.int32())

        res = exactly_one_equality_filter(
            p,
            [
                Filter("test", "=", 5956),
                Filter("test", ">=", 20),
                Filter("test2", "<=", "2024-04-06"),
            ],
        )

        assert res is None

    def test_exactly_one_equality_filter_no_filters(self):
        p = ("test", pa.int32())

        res = exactly_one_equality_filter(p, [])  # type: ignore

        assert res is None
