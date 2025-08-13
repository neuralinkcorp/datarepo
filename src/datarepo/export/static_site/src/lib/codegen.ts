import { ExportedCatalog, ExportedDatabase, ExportedTable, ExportedTablePartition } from './types';

enum BracketType {
  Parentheses,
  Brackets,
  Braces,
}

function openBracket(bracketType: BracketType) {
  switch (bracketType) {
    case BracketType.Parentheses:
      return '('
    case BracketType.Brackets:
      return '['
    case BracketType.Braces:
      return '{'
  }
}
function closeBracket(bracketType: BracketType) {
  switch (bracketType) {
    case BracketType.Parentheses:
      return ')'
    case BracketType.Brackets:
      return ']'
    case BracketType.Braces:
      return '}'
  }
}

function indent(code: string, spaces: number): string {
  return code.split('\n').map((line) => ' '.repeat(spaces) + line).join('\n')
}

function formatMultiLineArgs(filters: string[], bracket: BracketType): string {
  return openBracket(bracket) + '\n' + filters.map((filter => indent(filter, 4) + ',')).join('\n') + '\n' + closeBracket(bracket)
}

function formatPythonTupleOrParams(params: string[]): string {
  if (params.length <= 1) {
    return '(' + params.join(', ') + ')'
  } else {
    return formatMultiLineArgs(params, BracketType.Parentheses)
  }
}

function isStringPartition(partition: ExportedTablePartition): boolean {
  return partition.type_annotation === 'str' || partition.type_annotation === 'string';
}

interface GenTableCodeOptions {
  catalog: ExportedCatalog;
  database: ExportedDatabase;
  table: ExportedTable;

  /**
   * If true, the filters will be formatted as a SQL string.
   * Otherwise, they will be formatted as a list of datarepo `Filter` objects.
   */
  formatSqlFilter?: boolean;
}

export function genTableCode({ catalog, database, table, formatSqlFilter }: GenTableCodeOptions): string {
  const params = [`"${table.name}"`]

  if (table.partitions.length !== 0) {
    if (formatSqlFilter) {
      const stringFilter = table.partitions.map((partition) => {
        const partitionValue = isStringPartition(partition) ? `'${partition.value}'` : partition.value;
        return `${partition.column_name} = ${partitionValue}`
      }).join(' and ');
      params.push(`filters="${stringFilter}"`);
    } else {
      const filters = []

      for (const partition of table.partitions) {
        const value = isStringPartition(partition) ? `"${partition.value}"` : partition.value
        filters.push(`Filter("${partition.column_name}", "=", ${value})`)
      }

      /*
        We need to use formatMultiLineArgs for filters,
        as we want a hanging comma even when there is a
        single filter. Otherwise, python will break
        the named tuple into an array.
        e.g.
          (
                Filter('implant_id', '==', 4595),
          ),
      */
      params.push(`${formatMultiLineArgs(filters, BracketType.Parentheses)}`)
    }
  }

  if (table.selected_columns != null) {
    params.push(`columns=${formatMultiLineArgs(table.selected_columns.map((column => '"' + column + '"')), BracketType.Brackets)}`)
  }

  const formattedParams = formatPythonTupleOrParams(params)

  let retTable = `from ${catalog.package_name || 'datarepo_catalogs'} import ${catalog.name}\n`

  retTable += `from datarepo.core import Filter\n`

  retTable += `\n`
  retTable += `df = ${catalog.name}.db("${database.name}").table${formattedParams}\n`
  retTable += `print(df.collect())`

  return retTable.trim()
}
