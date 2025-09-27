# type: ignore
import unittest
from unittest.mock import ANY, patch

from polars import testing as pl_testing
import pytest

from datarepo.core import ModuleDatabase
from datarepo.core.catalog import Catalog
from test.functional.data import database, database2
from test.functional.data.database2 import frame3
from test.functional.data.database import frame1, frame2


class TestCore(unittest.TestCase):
    def setUp(self):
        dbs = {
            "database": ModuleDatabase(database),
            "database2": ModuleDatabase(database2),
        }

        self.catalog = Catalog(dbs)

    def test_get_dbs(self):
        assert self.catalog.dbs() == ["database", "database2"]

    def test_get_tables(self):
        db_1_tables = self.catalog.db("database").tables()

        assert set(db_1_tables) == set(
            [
                "my_parquet_table_hive",
                "my_parquet_table",
                "new_table",
                "new_table_2",
                "my_delta_table",
            ]
        )

        db_1_tables_with_deprecated = self.catalog.db("database").tables(
            show_deprecated=True
        )
        assert set(db_1_tables_with_deprecated) == set(
            [
                "my_parquet_table_hive",
                "my_parquet_table",
                "new_table",
                "new_table_2",
                "my_delta_table",
                "deprecated_table",
            ]
        )

        db_2_tables = self.catalog.db("database2").tables()

        assert db_2_tables == ["new_table_3"]

    def test_unregistered_db(self):
        with pytest.raises(KeyError) as exc_info:
            self.catalog.db("not_a_database")

        assert "Database 'not_a_database' not found" in str(exc_info.value)

    def test_unregistered_table(self):
        with pytest.raises(KeyError) as exc_info:
            self.catalog.db("database").table("not_a_table")

        assert "Table 'not_a_table' not found in database" in str(exc_info.value)

    def test_get_table(self):
        db_1 = self.catalog.db("database")
        frame = db_1.table("new_table")

        pl_testing.assert_frame_equal(frame, frame1)

        frame = db_1.table("new_table_2")

        pl_testing.assert_frame_equal(frame, frame2)

        db_2 = self.catalog.db("database2")
        frame = db_2.table("new_table_3")

        pl_testing.assert_frame_equal(frame, frame3)

    def test_get_deprecated_table(self):
        with patch("warnings.warn") as warn_patch:
            db_1 = self.catalog.db("database")
            frame = db_1.table("deprecated_table")

            # Assert a warning should have been shown
            warn_patch.assert_called_once_with(ANY, DeprecationWarning)

            pl_testing.assert_frame_equal(frame, frame1)
