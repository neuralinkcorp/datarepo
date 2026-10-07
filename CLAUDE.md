# CLAUDE.md

Guidance for Claude Code and other AI assistants working in this repository.

## What this is

`datarepo` is a declarative query layer over data that already exists somewhere else. You
describe a catalog of databases and tables in Python; `datarepo` resolves reads against the
underlying store and can export the catalog as a static site or a read-only ROAPI config.
There is no server and no state of its own — every table is a description of data that lives
in S3, ClickHouse, or behind a Python function.

That shapes most decisions here: a table definition is closer to a schema declaration than to
code that owns data, and the declared schema is deliberately allowed to differ from the
physical one (see below).

## Layout

| Path | Contents |
| :--- | :--- |
| `src/datarepo/core/catalog/` | `Catalog`, `Database`, `ModuleDatabase`, `DatabaseWithGlobalArgs`. |
| `src/datarepo/core/tables/` | One module per backend — `deltalake_table.py`, `parquet_table.py`, `clickhouse_table.py` — plus `decorator.py` for `@table`-defined Python functions. `metadata.py` holds `TableProtocol`, `TableMetadata`, `TableSchema`. |
| `src/datarepo/core/dataframe/` | `NlkDataFrame`, the lazy frame every table read returns. |
| `src/datarepo/export/` | `web.py` and `roapi.py` build the catalog site and the ROAPI config; `static_site/` is a Node app built by a hatch build hook. |
| `test/` | pytest, mirroring `src/` (`test/tables/`, `test/export/`). |
| `docs/` | mkdocs sources, published to data-repo.io. `docs/README.md` is also the PyPI readme. |

Every table type implements `TableProtocol`: a `table_metadata` attribute, `__call__(**kwargs)
-> NlkDataFrame`, and `get_schema() -> TableSchema`. Anything the catalog or the exporters need
should go through that protocol rather than special-casing a backend.

## Checks

CI is `.github/workflows/build_and_publish.yml`, on Python 3.10. Run all four locally before
proposing a change — they are quick and they are exactly what the workflow runs:

```sh
uv venv --python 3.10 .venv && . .venv/bin/activate
uv pip install -e ".[dev]"

pytest test --cov=src/datarepo --cov-report=term-missing
black --check src/datarepo
flake8 src/datarepo --count --select=E9,F63,F7,F82 --show-source --statistics
mypy src/datarepo
```

`main` is green on all four, so any failure is yours. Note that only `src/datarepo` is linted
and type-checked — `test/` is not — and that `flake8`'s second invocation in CI runs with
`--exit-zero`, so only the `E9,F63,F7,F82` set above can actually fail the build.

Dependencies are pinned for real reasons, recorded in `pyproject.toml`: `deltalake<1` pending
the delta-rs 1.0 upgrade guide, and `polars<1.14` because 1.15 broke `hive_partitioning` and
1.14 has an unresolved CI failure. Don't relax those bounds casually.

## Reading table code

`DeltalakeTable` is the most intricate backend and the one most often changed.

- **The declared schema wins.** `construct_df` reads with `self.schema`, not the physical Delta
  schema, and `_normalize_df` adds missing columns, casts, and reorders to match it. A column in
  the declared schema that is absent from the data comes back as nulls rather than an error.
  This is intentional, and it is why schema and storage can drift apart.
- **Reads go through DataFusion SQL**, not `to_pyarrow_dataset`. `construct_df` builds a
  `SELECT ... FROM "<name>" WHERE <predicate>` string and runs it through `QueryBuilder`, so the
  table's `name` is a SQL identifier, and filters become a predicate string via
  `datafusion_predicate_from_filters`.
- **`unique_columns` deduplicates with `.filter(pl.struct(...).is_last_distinct())`**, not
  `.unique()`, to dodge a polars bug on very large frames — and it keeps *later* rows, which
  matters when a backfill has rewritten earlier ones. The comment in place explains it; leave it.
- **`extra_cols` are computed, not read.** They are excluded from the columns requested from
  storage and added afterwards with `with_columns`.
- **Delta has its own versions.** `delta_table()` already accepts a `version` for transaction-log
  time travel. Keep that concept distinct from anything that versions a table's *schema* — the
  two are easy to conflate and the distinction has come up in review (#42, #46).

`get_schema()` prefers the `partition_columns` passed to the constructor over querying the remote
table, so catalog export works for a table that does not exist yet. Preserve that property: it is
why export does not need live credentials.

## Conventions

- Type annotations throughout, and Google-style docstrings with `Args:`/`Returns:` on public
  methods. mypy runs without `--strict` but does run clean. `deltalake_table.py` and
  `clickhouse_table.py` open with `from __future__ import annotations`; the other table modules
  do not, so follow the file you are in rather than adding it everywhere.
- `black` with the default 88-column line length owns formatting; don't hand-format around it.
- Keep a non-obvious workaround's comment attached to it. Several of them (the polars dedup, the
  eager-parquet read in `fetch_df_by_partition`, the `_last_checkpoint` cache duration) encode
  benchmark results or upstream bug numbers that are expensive to rediscover.
- `test/tables/test_deltalake_table.py` mixes both styles deliberately: `MagicMock` and
  `patch("polars.read_parquet", ...)` for the file-fetching helpers, and real Delta tables
  created under `tmp_path` with `DeltaTable.create` / `write_deltalake` for anything that
  exercises reads end to end. Prefer the real tables when the behaviour under test is schema
  normalisation or query construction — mocks there tend to assert the mock.

## Before opening a pull request

Check the issue for existing work first. Several issues here already have open PRs — #42 has
three — so the useful contribution is often a review or a rebase rather than another
implementation. Where a maintainer has asked a design question on an issue, answer it before
writing code against an assumed answer.
