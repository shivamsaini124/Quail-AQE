from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parser import QueryIR

BYPASS_QUERY_TYPES = {"INSERT", "CREATE", "UPDATE", "DELETE", "DROP", "ALTER", "MERGE"}


def should_approximate(ir: "QueryIR") -> tuple[bool, str]:
    query_type = ir.query_type.upper()
    if query_type in BYPASS_QUERY_TYPES:
        return False, "ddl_dml"

    if ir.has_distinct:                         
        return False, "distinct"

    if not ir.aggregations:
        return False, "no_aggregation"

    if ir.has_subquery and len(ir.tables) > 3:
        return False, "complex_subquery"

    return True, "ok"