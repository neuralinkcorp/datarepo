# Changelog

## 0.1.0

- Switch the ClickHouse client to `clickhouse-connect`.
- Default ClickHouse HTTP port is 8123 with TLS off. Pass `port=8443, secure=True` for HTTPS.
- `TableColumn` uses `name` instead of `column`.
- Delta unique-column dedup keeps the last distinct row (`is_last_distinct`) instead of `unique()`.
- ROAPI export no longer includes ClickHouse tables (unsupported format).
- ClickHouse queries use the table's stored config; SQL uses `config.table_name` or `self.name`.
- Web catalog export uses `partition_columns` when set, otherwise infers partitions from `docs_filters`.
