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

BYPASS_QUERY_TYPES = {"Insert", "Create", "Update", "Delete", "Drop", "Alter", "Merge"}

def should_approximate(ir: QueryIR) -> tuple[bool, str]:
    if ir.query_type in BYPASS_QUERY_TYPES:
        return False, "ddl_dml"
    if not ir.aggregations:
        return False, "no_aggregation"   # nothing to approximate
    if ir.has_distinct:
        return False, "distinct"         # DISTINCT + approximation = wrong cardinality
    if ir.has_subquery and len(ir.tables) > 3:
        return False, "complex_subquery" # too risky to rewrite safely
    return True, "ok"

@dataclass
class Phase1Features:
    num_filters: int
    num_joins: int
    has_groupby: bool
    num_aggregations: int
    has_subquery: bool
    join_types: list[str]    # INNER, LEFT, CROSS — CROSS joins are dangerous to sample over

def extract_phase1(ir: QueryIR) -> Phase1Features:
    return Phase1Features(
        num_filters      = len(ir.predicates),
        num_joins        = len(ir.joins),
        has_groupby      = bool(ir.groupby_cols),
        num_aggregations = len(ir.aggregations),
        has_subquery     = ir.has_subquery,
        join_types       = [j["type"] for j in ir.joins]
    )

@dataclass
class Phase2Features:
    selectivity: float       # fraction of rows estimated to pass WHERE
    variance: float          # population variance of the groupby/target column
    group_count: int         # estimated NDV of groupby key
    table_row_count: int     # from pg_class.reltuples or equivalent

def extract_phase2(ir: QueryIR, catalog: CatalogClient) -> Phase2Features:
    stats = catalog.get_stats(ir.tables[0])   # primary table
    sel   = _estimate_selectivity(ir.predicates, stats)
    return Phase2Features(
        selectivity     = sel,
        variance        = stats.column_variance(ir.groupby_cols),
        group_count     = stats.ndistinct(ir.groupby_cols),
        table_row_count = stats.row_count
    )

@dataclass
class SamplingPlan:
    strategy: str            # "sampling" | "filter_sampling" | "hybrid" | "exact"
    sample_rate: float
    stratify_on: list[str] | None
    predicted_error: float

def classify(p1: Phase1Features, p2: Phase2Features) -> SamplingPlan:

    # ── Tier 1: Rule-based (hard thresholds from your table) ──
    if p2.selectivity < 0.01:
        return SamplingPlan("exact", 1.0, None, 0.0)

    if p1.num_joins == 0 and p1.num_filters == 0 and not p1.has_groupby:
        return SamplingPlan("sampling", rate=0.1, stratify_on=None, predicted_error=0.05)

    if p1.num_filters >= 1 and p1.num_joins <= 1 and not p1.has_groupby:
        if p2.selectivity >= 0.1 and _is_low_variance(p2.variance):
            rate = _neyman_rate(p2.variance, p2.group_count, target_error=0.05)
            return SamplingPlan("filter_sampling", rate, stratify_on=p1.groupby_cols or None, predicted_error=0.05)

    if (p1.num_filters > 3 or p1.num_joins >= 2 or p1.has_groupby) and \
       0.01 <= p2.selectivity < 0.1:
        return SamplingPlan("hybrid", rate=None, stratify_on=p1.groupby_cols, predicted_error=None)
        # rate=None means ML model fills it in

    # ── Tier 2: Heuristic fallback ──
    if _confidence_heuristic(p1, p2) > 0.7:
        return SamplingPlan("sampling", rate=0.15, stratify_on=None, predicted_error=0.08)

    # ── Tier 3: ML model ──
    return ml_model.predict(p1, p2)   # trained XGBoost, returns full SamplingPlan


def _neyman_rate(variance: float, group_count: int, target_error: float) -> float:
    import math
    z = 1.96   # 95% CI
    n = (z ** 2 * variance) / (target_error ** 2)
    return min(n / max(group_count, 1), 1.0)

import sqlglot.optimizer as opt

def pushdown_and_prune(ir: QueryIR, catalog: CatalogClient) -> QueryIR:
    # Push WHERE predicates as close to the scan as possible
    ir.ast = opt.pushdown_predicates.pushdown_predicates(ir.ast)

    # Partition pruning: if a predicate matches the partition column,
    # rewrite the table scan to only touch matching partitions
    for table in ir.tables:
        part_col = catalog.partition_column(table)
        if not part_col:
            continue
        matching_preds = [p for p in ir.predicates if p["col"] == part_col]
        if matching_preds:
            ir.ast = _inject_partition_filter(ir.ast, table, matching_preds)

    return ir

import xgboost as xgb
import numpy as np

class HybridRatePredictor:
    def __init__(self, model_path: str):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict(self, p1: Phase1Features, p2: Phase2Features) -> tuple[float, float]:
        feature_vec = np.array([[
            p1.num_filters, p2.selectivity, p1.num_joins,
            int(p1.has_groupby), p2.variance, p2.group_count
        ]])
        dmat = xgb.DMatrix(feature_vec)
        preds = self.model.predict(dmat)
        return float(preds[0, 0]), float(preds[0, 1])  # sample_rate, predicted_error


