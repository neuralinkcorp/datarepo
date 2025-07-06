from typing import Any, Literal, Sequence, cast, NamedTuple

FilterOperator = Literal[
    "=",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "in",
    "not in",
    "contains",
    "includes",
    "includes",
    "includes any",
    "includes all",
]


class Filter(NamedTuple):
    """Filter represents a condition to be applied to a column in a table."""

    column: str
    operator: FilterOperator
    value: Any


InputFilters = Sequence[Filter] | Sequence[Sequence[Filter]]
NormalizedFilters = list[list[Filter]]


def normalize_filters(filters: InputFilters | None) -> NormalizedFilters:
    """
    Normalize filters to the standard format of a list of lists of filters, where each
    inner list represents a conjunction (AND) of filters and the outer list represents
    a disjunction (OR) of the inner conjunctions.

    We make a special case for the empty filter list, which is treated as no filters.
    Typically, a disjunction of no expressions should be false, but the user likely
    intends to apply no filters, and [] is easier to work with than [[]].
    """

    if filters is None or len(filters) == 0:
        return []
    elif all(isinstance(f, Filter) for f in filters):
        filters = cast(Sequence[Filter], filters)
        return [list(filters)]
    else:
        filters = cast(Sequence[Sequence[Filter]], filters)
        return [list(f) for f in filters]
