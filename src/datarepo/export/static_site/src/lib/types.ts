export interface ExportedTablePartition {
  column_name: string
  type_annotation: string | null
  value: string | number
}

export interface ExportedTableColumn {
  name: string
  type: string
  readonly?: boolean;
  filter_only?: boolean;
  has_stats?: boolean;
}

export interface ExportedTable {
  name: string
  description: string
  partitions: ExportedTablePartition[]
  columns: ExportedTableColumn[] | null
  selected_columns: string[] | null
  supports_sql_filter: boolean;
  table_type: 'FUNCTION' | 'DELTA_LAKE' | 'PARQUET';
  latency_info: string | null;
  example_notebook: string | null;
  data_input: string | null;
}

export interface ExportedDatabase {
  name: string
  tables: ExportedTable[]
}

export interface ExportedCatalogMetadata {
  jupyterhub_url?: string | null
}

export interface ExportedCatalog {
  name: string
  metadata: ExportedCatalogMetadata | null
  databases: ExportedDatabase[]
}

export interface ExportedDatarepo {
  catalogs: ExportedCatalog[]
}
