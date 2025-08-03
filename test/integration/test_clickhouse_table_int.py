import os
import pytest
import polars as pl
import pyarrow as pa
from typing import Generator

import clickhouse_driver
from testcontainers.clickhouse import ClickHouseContainer
from datarepo.core.tables.clickhouse_table import ClickHouseTable, ClickHouseTableConfig
from datarepo.core.tables.filters import Filter


@pytest.fixture(scope="module")
def clickhouse_container() -> Generator[ClickHouseContainer, None, None]:
    """Start a ClickHouse container for testing."""
    with ClickHouseContainer() as container:
        container.start()
        client = clickhouse_driver.Client.from_url(container.get_connection_url())
        client.execute("""
        CREATE TABLE IF NOT EXISTS default.test_table (
            implant_id Int64,
            date String,
            value Int64,
            str_value String
        ) ENGINE = MergeTree()
        ORDER BY (implant_id, date)
        """)
        
        # Insert test data
        client.execute("""
        INSERT INTO default.test_table 
        (implant_id, date, value, str_value)
        VALUES 
        (1, '2023-01-01', 100, 'alpha'),
        (1, '2023-01-02', 110, 'beta'),
        (2, '2023-01-01', 200, 'gamma'),
        (2, '2023-01-02', 210, 'delta'),
        (3, '2023-01-03', 300, 'epsilon')
        """)
        
        yield container


@pytest.fixture(scope="module")
def clickhouse_config(clickhouse_container: ClickHouseContainer) -> ClickHouseTableConfig:
    """Create a ClickHouseTableConfig for the container."""
    host = clickhouse_container.get_container_host_ip()
    port = clickhouse_container.get_exposed_port(9000)
    
    return ClickHouseTableConfig(
        host=host,
        port=int(port), 
        username="default",
        password="",
        database="default",
    )


@pytest.fixture(scope="module")
def clickhouse_table(clickhouse_config: ClickHouseTableConfig) -> ClickHouseTable:
    """Create a ClickHouseTable for testing."""
    return ClickHouseTable(
        name="test_table",
        schema=pa.schema([
            ("implant_id", pa.int64()),
            ("date", pa.string()),
            ("value", pa.int64()),
            ("str_value", pa.string()),
        ]),
        config=clickhouse_config,
        description="Test ClickHouse table",
    )


class TestClickHouseTableIntegration:
    def test_query_all_data(self, clickhouse_table: ClickHouseTable):
        """Test querying all data from the table."""
        df = clickhouse_table().collect()
        
        assert df.height == 5
        assert df.width == 4
        
        assert set(df.columns) == {"implant_id", "date", "value", "str_value"}
        
        assert df["implant_id"].to_list() == [1, 1, 2, 2, 3]
    
    def test_filter_by_equality(self, clickhouse_table: ClickHouseTable):
        """Test filtering by equality."""
        df = clickhouse_table(filters=[Filter("implant_id", "=", 1)]).collect()
        
        assert df.height == 2
        assert all(id == 1 for id in df["implant_id"])
    
    def test_filter_by_comparison(self, clickhouse_table: ClickHouseTable):
        """Test filtering by comparison."""
        df = clickhouse_table(filters=[Filter("value", ">", 200)]).collect()
        
        assert df.height == 2
        assert all(val > 200 for val in df["value"])
    
    def test_filter_by_in(self, clickhouse_table: ClickHouseTable):
        """Test filtering using IN operator."""
        df = clickhouse_table(filters=[Filter("implant_id", "in", [1, 3])]).collect()
        
        assert df.height == 3
        assert set(df["implant_id"].unique().to_list()) == {1, 3}
    
    def test_select_columns(self, clickhouse_table: ClickHouseTable):
        """Test selecting specific columns."""
        df = clickhouse_table(columns=["implant_id", "value"]).collect()
        
        assert df.width == 2
        assert set(df.columns) == {"implant_id", "value"}
    
    def test_combined_filters(self, clickhouse_table: ClickHouseTable):
        """Test combining multiple filters."""
        df = clickhouse_table(filters=[[Filter("implant_id", "=", 1)], 
                                     [Filter("date", "=", "2023-01-03")]]).collect()
        
        assert df.height == 3
        
        df = clickhouse_table(filters=[[Filter("implant_id", "=", 1), 
                                     Filter("date", "=", "2023-01-01")]]).collect()
        
        assert df.height == 1
        assert df["implant_id"][0] == 1
        assert df["date"][0] == "2023-01-01"
    
    def test_string_like_filter(self, clickhouse_table: ClickHouseTable):
        """Test LIKE filter for string matching."""
        df = clickhouse_table(filters=[Filter("str_value", "contains", "%lta%")]).collect()
        
        assert df.height == 1
        assert df["str_value"][0] == "delta"