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

// Escapes a value for use inside a single-quoted SQL string literal by doubling embedded
// single quotes (the standard SQL escape), independent of the surrounding Python string.
function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''")
}

// Escapes a value for use as a LIKE pattern: '%' and '_' are wildcards to LIKE itself, so a
// literal value containing them (e.g. a filter value of "50%") must have them escaped, in
// addition to the usual SQL string-literal quote escaping.
function escapeSqlLikeValue(value: string): string {
  return escapeSqlString(value.replace(/\\/g, '\\\\').replace(/%/g, '\\%').replace(/_/g, '\\_'))
}

// Renders a single scalar (non-array) partition value as a SQL literal, quoting and escaping
// it when it is a string regardless of what type_annotation claims (e.g. "large_string" or a
// date-typed string column still need quotes).
function formatSqlScalar(value: unknown, isString: boolean): string {
  if (typeof value === 'string' || isString) {
    return `'${escapeSqlString(String(value))}'`
  }
  return `${value}`
}

function formatSqlPredicate(partition: ExportedTablePartition): string {
  const operator = partitionOperator(partition)
  if (operator === 'is null' || operator === 'is not null') {
    return `${partition.column_name} ${operator}`
  }
  if (operator === 'contains') {
    return `${partition.column_name} like '%${escapeSqlLikeValue(String(partition.value))}%' escape '\\'`
  }
  if (operator === 'in' || operator === 'not in') {
    const values = Array.isArray(partition.value) ? partition.value : [partition.value]
    const rendered = values.map((v) => formatSqlScalar(v, isStringPartition(partition)))
    return `${partition.column_name} ${operator} (${rendered.join(', ')})`
  }
  const value = formatSqlScalar(partition.value, isStringPartition(partition))
  return `${partition.column_name} ${operator} ${value}`
}

function formatPythonLiteral(value: ExportedFilterValue): string | null {
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
  if (Array.isArray(value)) {
    const values = value.map(formatPythonLiteral)
    if (values.some(value => value === null)) return null
    return '[' + values.join(', ') + ']'
  }
  return null
}

function formatFilterValue(partition: ExportedTablePartition): string | null {
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
  const params = [JSON.stringify(table.name)]

  if (table.partitions.length !== 0) {
    if (formatSqlFilter) {
      const stringFilter = table.partitions.map(formatSqlPredicate).join(' and ');
      // JSON string escaping produces a valid Python double-quoted string literal for any
      // value (including one containing embedded '"' or '\', which would otherwise end the
      // Python string early), since Python accepts the same \", \\, and \uXXXX escapes JSON does.
      params.push(`filters=${JSON.stringify(stringFilter)}`);
    } else {
      const filters = []

      for (const partition of table.partitions) {
        const value = formatFilterValue(partition)
        if (value === null) return '# cannot render this filter value'
        filters.push(
          `Filter(${JSON.stringify(partition.column_name)}, ${JSON.stringify(partitionOperator(partition))}, ${value})`
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
    params.push(`columns=${formatMultiLineArgs(table.selected_columns.map(column => JSON.stringify(column)), BracketType.Brackets)}`)
  }

  const formattedParams = formatPythonTupleOrParams(params)

  let retTable = `from ${catalog.package_name || 'datarepo_catalogs'} import ${catalog.name}\n`

  retTable += `from datarepo.core import Filter\n`

  retTable += `\n`
  retTable += `df = ${catalog.name}.db(${JSON.stringify(database.name)}).table${formattedParams}.collect()`

  return retTable.trim()
}
