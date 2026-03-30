import sqlglot
import sqlglot.expressions as exp
from dataclasses import dataclass

@dataclass
class QueryIR:
    query_type: str
    tables: list[str]
    predicates: list[dict]
    joins: list[dict]
    groupby_cols: list[str]
    aggregations: list[str]
    has_subquery: bool
    has_order_by: bool
    has_distinct: bool
    ast: exp.Expression

def parse(sql: str, dialect: str = "postgres") -> QueryIR:
    ast = sqlglot.parse_one(sql, dialect=dialect)
    return QueryIR(
        query_type   = type(ast).__name__,
        tables       = [t.name for t in ast.find_all(exp.Table)],
        predicates   = _extract_predicates(ast),
        joins        = _extract_joins(ast),
        groupby_cols = _extract_groupby_cols(ast),
        aggregations = _extract_aggregations(ast),
        has_subquery = bool(ast.find(exp.Subquery)),
        has_order_by = bool(ast.find(exp.Order)),
        has_distinct = bool(ast.find(exp.Distinct)),
        ast          = ast
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
    
    # the first table in FROM is always the left side of the first join
    from_clause = ast.find(exp.From)
    left_table = from_clause.find(exp.Table).name if from_clause else ""

    for join in ast.find_all(exp.Join):
        right_table = join.alias_or_name
        joins.append({
            "type": join.side or "INNER",
            "left": left_table,
            "right": right_table,
            "on_col": str(join.args.get("on", "")),
        })
        # each join's right side becomes the next join's left side
        left_table = right_table

    return joins

def _extract_groupby_cols(ast: exp.Expression) -> list[str]:
    group = ast.find(exp.Group)
    if not group:
        return []
    return [col.name for col in group.find_all(exp.Column)]

def _extract_aggregations(ast: exp.Expression) -> list[str]:
    aggs = []
    for node in ast.find_all(exp.Count, exp.Sum, exp.Avg, exp.Max, exp.Min):
        aggs.append(type(node).__name__.upper())
    return aggs