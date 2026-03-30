import sqlglot
import sqlglot.expressions as exp
from dataclasses import dataclass, field

@dataclass
class QueryIR:
    query_type: str                  # "Select", "Insert", "Create", etc.
    tables: list[str]
    predicates: list[dict]           # [{col, op, val, is_range}]
    joins: list[dict]                # [{type, left, right, on_col}]
    groupby_cols: list[str]
    aggregations: list[str]          # ["COUNT", "SUM", ...]
    has_subquery: bool
    has_order_by: bool
    has_distinct: bool
    ast: exp.Expression              # kept for rewriter in stage 7

def parse(sql: str, dialect: str = "postgres") -> QueryIR:
    ast = sqlglot.parse_one(sql, dialect=dialect)
    return QueryIR(
        query_type = type(ast).__name__,
        tables     = [t.name for t in ast.find_all(exp.Table)],
        predicates = _extract_predicates(ast),
        joins      = _extract_joins(ast),
        groupby_cols = [c.name for c in ast.find_all(exp.Group)],
        aggregations = [f.name.upper() for f in ast.find_all(exp.Anonymous, exp.Count, exp.Sum, exp.Avg)],
        has_subquery  = bool(ast.find(exp.Subquery)),
        has_order_by  = bool(ast.find(exp.Order)),
        has_distinct  = bool(ast.find(exp.Distinct)),
        ast = ast
    )

def _extract_predicates(ast: exp.Expression) -> list[dict]:
    predicates = []
    where = ast.find(exp.Where)
    if not where:
        return predicates
 
    for cond in where.find_all(exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ):
        col = cond.left.name if hasattr(cond.left, "name") else str(cond.left)
        predicates.append({
            "col": col,
            "op": type(cond).__name__,
            "val": str(cond.right),
            "is_range": isinstance(cond, (exp.Between,)),
        })
    return predicates

def _extract_joins(ast: exp.Expression) -> list[dict]:
    joins = []
    for join in ast.find_all(exp.Join):
        joins.append({
            "type": join.side or "INNER",          # side: LEFT, RIGHT, FULL, or None→INNER
            "left": join.this.name if hasattr(join.this, "name") else str(join.this),
            "right": join.alias_or_name,
            "on_col": str(join.args.get("on", "")),
        })
    return joins