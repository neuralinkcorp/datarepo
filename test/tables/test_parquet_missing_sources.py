from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import boto3
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import datarepo.core as core
from datarepo.core import tables
from datarepo.core.tables import Filter, ParquetTable, Partition, PartitioningScheme


@pytest.fixture(autouse=True)
def isolated_aws_environment(monkeypatch, tmp_path):
    for name in os.environ:
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fixture")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fixture")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "unused-config"))
    monkeypatch.setenv(
        "AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "unused-credentials")
    )


@pytest.fixture
def empty_partition(tmp_path, request):
    scheme = request.param
    directory = tmp_path / ("group=42" if scheme == PartitioningScheme.HIVE else "42")
    directory.mkdir()
    return (
        ParquetTable(
            name="signals",
            uri=str(tmp_path),
            partitioning=[Partition("group", pl.Int32)],
            partitioning_scheme=scheme,
        ),
        directory,
    )


@pytest.mark.parametrize("empty_partition", list(PartitioningScheme), indirect=True)
def test_empty_partition_reports_table_and_resolved_source(empty_partition):
    table, directory = empty_partition
    with pytest.raises(FileNotFoundError) as error:
        table(filters=[Filter("group", "=", 42)])

    assert type(error.value) is tables.DatasourceNotAvailable
    assert core.DatasourceNotAvailable is tables.DatasourceNotAvailable
    assert "signals" in str(error.value)
    assert str(directory) in str(error.value)
    assert isinstance(error.value.__cause__, pl.exceptions.ComputeError)


@pytest.mark.parametrize("empty_partition", list(PartitioningScheme), indirect=True)
@pytest.mark.parametrize("schema", [{}, {"value": pl.Int64}])
def test_empty_partition_with_schema_preserves_zero_rows(empty_partition, schema):
    table, _ = empty_partition
    original_schema = schema.copy()
    table.schema = schema
    expected_schema = {**schema, "group": pl.Int32}
    query = table(filters=[Filter("group", "=", 42)], columns=list(expected_schema))

    assert isinstance(query, pl.LazyFrame)
    assert_frame_equal(query.collect(), pl.DataFrame(schema=expected_schema))
    assert_frame_equal(
        query.sort("group").collect(), pl.DataFrame(schema=expected_schema)
    )
    assert schema == original_schema
    assert table.schema == original_schema


def test_empty_source_preserves_remaining_hive_partitions_and_filters(tmp_path):
    (tmp_path / "group=42").mkdir()
    table = ParquetTable(
        name="signals",
        uri=str(tmp_path),
        partitioning=[Partition("group", pl.Int32), Partition("date", pl.String)],
        partitioning_scheme=PartitioningScheme.HIVE,
        schema={"value": pl.Float64},
    )
    query = table(
        filters=[Filter("group", "=", 42), Filter("value", ">", 0)],
        columns=["date", "group", "value"],
    )
    assert_frame_equal(
        query.collect(),
        pl.DataFrame(
            schema={"date": pl.String, "group": pl.Int32, "value": pl.Float64}
        ),
    )
    assert table.schema == {"value": pl.Float64}


@pytest.mark.parametrize("schema", [None, {"value": pl.Int64}])
@pytest.mark.parametrize("empty_file", [False, True])
def test_existing_sources_can_return_zero_rows(tmp_path, schema, empty_file):
    source = pl.DataFrame({"value": [10, 20]})
    if empty_file:
        source = source.clear()
    source.write_parquet(tmp_path / "data.parquet")
    table = ParquetTable(
        name="signals", uri=str(tmp_path), partitioning=[], schema=schema
    )
    query = table(filters=[Filter("value", ">", 100)], columns=["value"])
    assert_frame_equal(query.collect(), source.clear())


@pytest.mark.parametrize("schema", [None, {"value": pl.Int64}])
def test_corrupt_source_is_not_missing_or_empty(tmp_path, schema):
    (tmp_path / "data.parquet").write_bytes(b"not a parquet file")
    table = ParquetTable(
        name="signals", uri=str(tmp_path), partitioning=[], schema=schema
    )
    with pytest.raises(pl.exceptions.ComputeError, match="parquet|Parquet"):
        table().collect()


@pytest.mark.parametrize("schema", [None, {"value": pl.Int64}])
def test_missing_literal_file_preserves_file_not_found(tmp_path, schema):
    table = ParquetTable(
        name="signals",
        uri=str(tmp_path / "absent.parquet"),
        partitioning=[],
        schema=schema,
    )
    with pytest.raises(FileNotFoundError) as error:
        table().collect()
    assert type(error.value) is FileNotFoundError
    assert "absent.parquet" in str(error.value)


@pytest.mark.parametrize("schema", [None, {}, {"value": pl.Int64}])
def test_empty_glob_without_partitions(tmp_path, schema):
    table = ParquetTable(
        name="signals", uri=str(tmp_path / "*.parquet"), partitioning=[], schema=schema
    )
    if schema is None:
        with pytest.raises(FileNotFoundError, match="signals"):
            table()
    else:
        assert_frame_equal(table().collect(), pl.DataFrame(schema=schema))


@pytest.mark.parametrize("schema", [None, {"value": pl.Int64}])
def test_unreadable_file_is_not_missing_or_empty(tmp_path, schema):
    file = tmp_path / "data.parquet"
    pl.DataFrame({"value": [10]}).write_parquet(file)
    file.chmod(0)
    try:
        if os.access(file, os.R_OK):
            pytest.skip("This platform or user can read files with mode 000")
        table = ParquetTable(
            name="signals", uri=str(file), partitioning=[], schema=schema
        )
        with pytest.raises(PermissionError):
            table().collect()
    finally:
        file.chmod(0o600)


def test_declared_schema_and_repeated_queries_are_preserved(tmp_path):
    (tmp_path / "42").mkdir()
    table = ParquetTable(
        name="signals",
        uri=str(tmp_path),
        partitioning=[Partition("group", pl.Int32)],
        schema={"value": pl.Int64, "optional": pl.String},
    )
    empty_query = table(filters=[Filter("group", "=", 42)])
    pl.DataFrame(
        {"value": [20, 10], "optional": [None, None]}, schema=table.schema
    ).write_parquet(tmp_path / "42" / "data.parquet")
    query = table(
        filters=[Filter("group", "=", 42)], columns=["group", "optional", "value"]
    )
    assert_frame_equal(
        query.sort("value").collect(),
        pl.DataFrame(
            {"group": [42, 42], "optional": [None, None], "value": [10, 20]},
            schema={"group": pl.Int32, "optional": pl.String, "value": pl.Int64},
        ),
    )
    assert empty_query.collect().height == 0
    assert table.schema == {"value": pl.Int64, "optional": pl.String}


@pytest.mark.parametrize("schema", [None, {"value": pl.Int64}])
def test_parquet_rows_are_still_read_lazily(tmp_path, schema):
    file = tmp_path / "data.parquet"
    pl.DataFrame({"value": list(range(100))}).write_parquet(
        file, compression="uncompressed"
    )
    data = bytearray(file.read_bytes())
    footer_start = len(data) - 8 - int.from_bytes(data[-8:-4], "little")
    # Keep the schema in the footer intact, but make the data pages unreadable.
    data[4:footer_start] = b"\x00" * (footer_start - 4)
    file.write_bytes(data)
    table = ParquetTable(name="signals", uri=str(file), partitioning=[], schema=schema)

    query = table(columns=["value"])
    assert isinstance(query, pl.LazyFrame)
    assert query.collect_schema() == {"value": pl.Int64}
    with pytest.raises(pl.exceptions.ComputeError):
        query.collect()


@pytest.fixture
def s3_listing():
    prefixes = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlsplit(self.path).query)
            if query.get("list-type") != ["2"]:
                self.send_error(400)
                return
            prefix = query.get("prefix", [""])[0]
            prefixes.append(prefix)
            if prefix.startswith("denied/"):
                status = 403
                body = b"<Error><Code>AccessDenied</Code><Message>Fixture denied</Message></Error>"
            else:
                status = 200
                body = (
                    b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                    b"<Name>fixture</Name><KeyCount>0</KeyCount>"
                    b"<IsTruncated>false</IsTruncated></ListBucketResult>"
                )
            self.send_response(status)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    session = boto3.Session(
        aws_access_key_id="fixture",
        aws_secret_access_key="fixture",
        region_name="us-east-1",
    )
    try:
        yield {
            "boto3_session": session,
            "endpoint_url": f"http://127.0.0.1:{server.server_port}",
        }, prefixes
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("schema", [None, {"value": pl.Int64}])
def test_empty_s3_prefix(s3_listing, schema):
    options, prefixes = s3_listing
    table = ParquetTable(
        name="signals",
        uri="s3://fixture/missing",
        partitioning=[Partition("group", pl.Int32)],
        schema=schema,
    )
    if schema is None:
        with pytest.raises(FileNotFoundError, match="s3://fixture/missing/42/"):
            table(filters=[Filter("group", "=", 42)], **options)
    else:
        query = table(filters=[Filter("group", "=", 42)], **options)
        assert_frame_equal(
            query.collect(), pl.DataFrame(schema={"value": pl.Int64, "group": pl.Int32})
        )
    assert prefixes and all(prefix == "missing/42/" for prefix in prefixes)


@pytest.mark.parametrize("schema", [None, {"value": pl.Int64}])
def test_s3_access_denied_is_not_missing_or_empty(s3_listing, schema):
    options, prefixes = s3_listing
    table = ParquetTable(
        name="signals", uri="s3://fixture/denied/", partitioning=[], schema=schema
    )
    with pytest.raises(pl.exceptions.ComputeError, match="403 Forbidden"):
        table(**options).collect()
    assert prefixes and all(prefix == "denied/" for prefix in prefixes)
