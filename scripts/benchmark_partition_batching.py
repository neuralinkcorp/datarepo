#!/usr/bin/env python3
"""Benchmark peak memory: batched vs unbatched partition file reads."""

from __future__ import annotations

import argparse
import tracemalloc

import polars as pl
import pyarrow as pa

from datarepo.core.tables.deltalake_table import (
    DEFAULT_PARTITION_BATCH_SIZE,
    fetch_dfs_by_paths,
    fetch_dfs_by_paths_batching,
)


def _allocating_read(_source: str, **_kwargs) -> pl.DataFrame:
    return pl.DataFrame({"payload": list(range(100_000))})


def peak_bytes_for(reader) -> int:
    tracemalloc.start()
    try:
        reader()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-count", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PARTITION_BATCH_SIZE)
    args = parser.parse_args()

    schema = pa.schema([("payload", pa.int64())])
    files = [f"synthetic/file-{i}.parquet" for i in range(args.file_count)]

    import polars as pl_module

    original = pl_module.read_parquet
    pl_module.read_parquet = _allocating_read  # type: ignore[assignment]
    try:
        unbatched = peak_bytes_for(lambda: fetch_dfs_by_paths(files, schema=schema))
        batched = peak_bytes_for(
            lambda: fetch_dfs_by_paths_batching(
                files, schema=schema, batch_size=args.batch_size
            )
        )
    finally:
        pl_module.read_parquet = original

    reduction = 100.0 * (1 - batched / unbatched) if unbatched else 0.0
    print(f"files={args.file_count} batch_size={args.batch_size}")
    print(f"peak_unbatched_bytes={unbatched}")
    print(f"peak_batched_bytes={batched}")
    print(f"peak_memory_reduction_pct={reduction:.1f}")


if __name__ == "__main__":
    main()
