from query_ir import QueryIR
BYPASS_QUERY_TYPES = {"Insert", "Create", "Update", "Delete", "Drop", "Alter", "Merge"}
 
def should_approximate(ir: QueryIR) -> tuple[bool, str]:
    if ir.query_type in BYPASS_QUERY_TYPES:
        return False, "ddl_dml"
    if not ir.aggregations:
        return False, "no_aggregation"
 
    if ir.has_distinct:
        return False, "distinct"
 
    if ir.has_subquery and len(ir.tables) > 3:
        return False, "complex_subquery"
 
    return True, "ok"