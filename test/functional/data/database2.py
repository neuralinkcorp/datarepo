import polars as pl

from datarepo.core import NlkDataFrame, table

frame3 = pl.LazyFrame({"a": ["x", "y", "z"]})


@table
def new_table_3() -> NlkDataFrame:
    return frame3
