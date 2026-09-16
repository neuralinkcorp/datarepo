"""Partition pruning must preserve disjunctions through real local Parquet reads."""

from collections import Counter
from copy import deepcopy
import operator
from pathlib import Path

import polars as pl
import pytest

from datarepo.core.tables import parquet_table as module
from datarepo.core.tables.filters import Filter
from datarepo.core.tables.parquet_table import ParquetTable
from datarepo.core.tables.util import Partition, PartitioningScheme


A = Filter("subject", "=", "A")
S1 = Filter("session", "=", 1)
HIGH = Filter("value", ">", 10)
ROWS = [
    ("A", 1, 1, 10),
    ("A", 1, 1, 10),  # Multiplicity must survive optimization.
    ("A", 1, 2, 20),
    ("A", 2, 3, 30),
    ("A0", 1, 4, 40),
    ("B", 1, 5, 50),
]
NAMES = ["subject", "session", "id", "value"]


@pytest.fixture(autouse=True)
def local_storage(monkeypatch):
    # Only credential lookup is replaced; Polars performs all reads and filtering.
    monkeypatch.setattr(module, "get_storage_options", lambda **kwargs: {})


@pytest.fixture
def directory_table(tmp_path):
    for subject, ids, values in (("A", [1, 2], [10, 20]), ("A0", [3], [30])):
        directory = tmp_path / subject
        directory.mkdir()
        pl.DataFrame({"id": ids, "value": values}).write_parquet(
            directory / "df.parquet"
        )
    return ParquetTable(
        "observations", str(tmp_path), [Partition("subject", pl.String)]
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_satisfied_branch_keeps_all_rows_in_selected_directory(
    directory_table, reverse
):
    filters = [[A], [A, HIGH]]
    if reverse:
        filters.reverse()
    before = deepcopy(filters)
    for _ in range(2):
        result = directory_table(
            filters=filters, columns=["value", "subject", "id"]
        ).collect()
        assert result.columns == ["value", "subject", "id"]
        assert result.sort("id").rows() == [(10, "A", 1), (20, "A", 2)]
        assert filters == before


@pytest.fixture
def hive_table(tmp_path):
    data = pl.DataFrame(ROWS, schema=NAMES, orient="row")
    for (subject, session), frame in data.group_by("subject", "session"):
        directory = tmp_path / f"subject={subject}" / f"session={session}"
        directory.mkdir(parents=True)
        frame.drop("subject", "session").write_parquet(directory / "df.parquet")
    return ParquetTable(
        "observations",
        str(tmp_path),
        [Partition("subject", pl.String), Partition("session", pl.Int64)],
        partitioning_scheme=PartitioningScheme.HIVE,
    )


def test_true_residual_stops_deeper_prefix_pruning(hive_table):
    filters = [[A], [A, S1, HIGH]]
    uri, partitions, residual, applied = hive_table._build_uri_from_filters(filters)
    assert uri == str(Path(hive_table.uri) / "subject=A") + "/"
    assert partitions == [Partition("session", pl.Int64)]
    assert residual == [[]]
    assert applied == [A]
    assert filters == [[A], [A, S1, HIGH]]


@pytest.mark.parametrize(
    "filters",
    [
        [[A], [A, HIGH]],
        [[A, HIGH], [A]],
        [[A, HIGH], [A, Filter("value", "<", 20)]],
        [[A], [Filter("subject", "=", "B")]],
        [[A, S1], [A, S1, HIGH]],
        [[A, S1, HIGH], [A, S1]],
        [[A], [A, S1]],
        [[A, S1]],
        [[A], [A]],
        [[A, Filter("value", "<", 0)]],
        [[A, Filter("subject", "=", "B")]],
        [[], [A, HIGH]],  # Caller-empty groups keep their existing convention.
        [[]],
        [],
        [[A, HIGH], [A, Filter("session", "=", 2)]],
        [[A, S1, HIGH], [A, S1, Filter("value", "<", 20)]],
    ],
    ids=[
        "true-first",
        "true-last",
        "two-residuals",
        "different-subjects",
        "two-partitions-true-first",
        "two-partitions-true-last",
        "do-not-overprune",
        "all-consumed",
        "identical-branches",
        "zero-rows",
        "contradiction",
        "caller-empty-branch",
        "only-empty-branch",
        "no-filters",
        "different-residual-columns",
        "two-partitions-two-residuals",
    ],
)
def test_optimized_reads_match_independent_row_evaluator(hive_table, filters):
    comparisons = {"=": operator.eq, ">": operator.gt, "<": operator.lt}
    nonempty = [branch for branch in filters if branch]
    expected = []
    for values in ROWS:
        row = dict(zip(NAMES, values))
        if not nonempty or any(
            all(comparisons[f.operator](row[f.column], f.value) for f in branch)
            for branch in nonempty
        ):
            expected.append((row["value"], row["subject"], row["session"], row["id"]))

    before = deepcopy(filters)
    for _ in range(2):
        actual = hive_table(
            filters=filters, columns=["value", "subject", "session", "id"]
        ).collect()
        assert actual.columns == ["value", "subject", "session", "id"]
        assert Counter(actual.rows()) == Counter(expected)
        assert filters == before


def test_caller_empty_branch_is_not_optimizer_created_truth(hive_table):
    uri, partitions, residual, applied = hive_table._build_uri_from_filters(
        [[], [A, HIGH]]
    )
    assert uri == hive_table.uri + "/"
    assert partitions == hive_table.partitioning
    assert residual == [[], [A, HIGH]]
    assert applied == []
