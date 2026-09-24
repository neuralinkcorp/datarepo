import { ExportedCatalog, ExportedDatabase, ExportedFilterValue, ExportedTable, ExportedTablePartition } from './types';

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

function partitionOperator(partition: ExportedTablePartition): string {
  return partition.operator || '='
}

function formatSqlPredicate(partition: ExportedTablePartition): string {
  const operator = partitionOperator(partition)
  if (operator === 'is null' || operator === 'is not null') {
    return `${partition.column_name} ${operator}`
  }
  if (operator === 'contains') {
    return `${partition.column_name} like '%${partition.value}%'`
  }
  const value = isStringPartition(partition) ? `'${partition.value}'` : partition.value
  return `${partition.column_name} ${operator} ${value}`
}

function formatPythonLiteral(value: ExportedFilterValue): string {
  if (value === null) return 'None'
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'number' && Number.isFinite(value)) {
    // JSON transport has already converted numbers to JavaScript precision.
    // An unsafe integer's shortest decimal may differ from its exact value:
    // keep a floating-point token so Python performs the same rounding.
    if (Number.isInteger(value) && !Number.isSafeInteger(value)) return value.toExponential()
    return Object.is(value, -0) ? '-0.0' : String(value)
  }
  if (Array.isArray(value)) return '[' + value.map(formatPythonLiteral).join(', ') + ']'
  throw new TypeError('Cannot generate a Python literal for this filter value')
}

function formatFilterValue(partition: ExportedTablePartition): string {
  const operator = partitionOperator(partition)
  if (
    operator === 'is null' ||
    operator === 'is not null' ||
    partition.value === null ||
    partition.value === undefined
  ) {
    return 'None'
  }
  return formatPythonLiteral(partition.value)
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
  const params = [formatPythonLiteral(table.name)]

  if (table.partitions.length !== 0) {
    if (formatSqlFilter) {
      const stringFilter = table.partitions.map(formatSqlPredicate).join(' and ');
      params.push(`filters="${stringFilter}"`);
    } else {
      const filters = []

      for (const partition of table.partitions) {
        filters.push(
          `Filter(${formatPythonLiteral(partition.column_name)}, ${formatPythonLiteral(partitionOperator(partition))}, ${formatFilterValue(partition)})`
        )
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
    params.push(`columns=${formatMultiLineArgs(table.selected_columns.map(formatPythonLiteral), BracketType.Brackets)}`)
  }

  const formattedParams = formatPythonTupleOrParams(params)

  let retTable = `from ${catalog.package_name || 'datarepo_catalogs'} import ${catalog.name}\n`

  retTable += `from datarepo.core import Filter\n`

  retTable += `\n`
  retTable += `df = ${catalog.name}.db(${formatPythonLiteral(database.name)}).table${formattedParams}.collect()`

  return retTable.trim()
}
