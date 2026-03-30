# aqe/rewriter/rewriter.py

from __future__ import annotations

import sqlglot
import sqlglot.expressions as exp
from dataclasses import dataclass

from parser.query_ir import QueryIR
from classifier.sampling_plan import SamplingPlan
from rewriter.sample_builder import SampleBuilder, SampleFragment, SampleSizeError


# ── Aggregates that must be scaled by 1/rate ─────────────────────────────────
# If you sample 10% and COUNT returns 500, the true estimate is 5000.
SCALE_UP_AGGS = {"COUNT", "SUM"}

# Aggregates that are rate-invariant — do NOT scale these.
# AVG, MIN, MAX are true statistics of the distribution; scaling distorts them.
# PERCENTILE and MEDIAN are also invariant for the same reason.
INVARIANT_AGGS = {"AVG", "MIN", "MAX", "MEDIAN", "PERCENTILE_CONT", "PERCENTILE_DISC"}

# COUNT DISTINCT needs a higher sample rate floor because accuracy degrades fast.
COUNT_DISTINCT_MIN_RATE = 0.30


@dataclass
class RewriteResult:
    sql:           str
    strategy:      str          # "uniform" | "stratified" | "hash" | "exact_fallback"
    sample_rate:   float
    is_approximate: bool
    warnings:      list[str]    # e.g. "COUNT DISTINCT detected, rate bumped to 0.30"


class QueryRewriter:
    """
    Orchestrates the full rewrite pipeline:

      1. Pre-flight checks  — detect COUNT DISTINCT, adjust rate if needed
      2. SampleBuilder      — produce the SampleFragment (CTE or TABLESAMPLE)
      3. AST injection      — splice the fragment into the query's FROM clause
      4. Aggregate scaling  — wrap COUNT/SUM with the inverse-rate multiplier
      5. CTE prepending     — attach the WITH clause to the final query
      6. SQL generation     — emit final SQL string via sqlglot
    """

    def __init__(self, builder: SampleBuilder | None = None, dialect: str = "postgres"):
        self.builder = builder or SampleBuilder()
        self.dialect = dialect

    # ── public entry point ────────────────────────────────────────────────────

    def rewrite(self, ir: QueryIR, plan: SamplingPlan) -> RewriteResult:
        warnings: list[str] = []

        # ── 1. Pre-flight: handle COUNT DISTINCT ──────────────────────────────
        plan = self._adjust_for_count_distinct(ir, plan, warnings)

        # ── 2. Build sample fragment ──────────────────────────────────────────
        try:
            fragment = self.builder.build(ir, plan)
        except SampleSizeError as e:
            # n too small → fall back to exact, return original SQL unchanged
            return RewriteResult(
                sql           = ir.original_sql,
                strategy      = "exact_fallback",
                sample_rate   = 1.0,
                is_approximate = False,
                warnings      = [str(e)],
            )

        # ── 3. Clone AST (never mutate the original) ──────────────────────────
        ast = ir.ast.copy()

        # ── 4. Inject sample into FROM clause ─────────────────────────────────
        ast = self._inject_sample(ast, ir.tables[0], fragment)

        # ── 5. Scale aggregates ───────────────────────────────────────────────
        ast = self._scale_aggregates(ast, fragment.sample_rate)

        # ── 6. Prepend CTE if stratified or hash ──────────────────────────────
        if fragment.cte_sql:
            ast = self._prepend_cte(ast, fragment.cte_sql)

        # ── 7. Emit SQL ───────────────────────────────────────────────────────
        final_sql = ast.sql(dialect=self.dialect, pretty=True)

        return RewriteResult(
            sql            = final_sql,
            strategy       = fragment.strategy,
            sample_rate    = fragment.sample_rate,
            is_approximate = True,
            warnings       = warnings,
        )

    # ── pre-flight ────────────────────────────────────────────────────────────

    def _adjust_for_count_distinct(
        self,
        ir: QueryIR,
        plan: SamplingPlan,
        warnings: list[str],
    ) -> SamplingPlan:
        """
        COUNT(DISTINCT col) accuracy degrades sharply below ~30% sample rate.
        If detected, bump rate to the floor and warn the caller.
        """
        has_count_distinct = any(
            isinstance(node, exp.Distinct)
            for node in ir.ast.find_all(exp.Distinct)
        )
        if has_count_distinct and plan.sample_rate < COUNT_DISTINCT_MIN_RATE:
            warnings.append(
                f"COUNT DISTINCT detected: sample rate bumped from "
                f"{plan.sample_rate:.2f} to {COUNT_DISTINCT_MIN_RATE:.2f}"
            )
            # Return a new plan with the adjusted rate; don't mutate original
            from dataclasses import replace
            return replace(plan, sample_rate=COUNT_DISTINCT_MIN_RATE)
        return plan

    # ── AST surgery ───────────────────────────────────────────────────────────

    def _inject_sample(
        self,
        ast: exp.Expression,
        original_table: str,
        fragment: SampleFragment,
    ) -> exp.Expression:
        """
        Finds the FROM clause table reference matching `original_table` and
        replaces it with the sample table_ref.

        For TABLESAMPLE (uniform), table_ref is:
            "orders TABLESAMPLE BERNOULLI (10.0)"
        For CTE-based (stratified/hash), table_ref is:
            "_aqe_strat WHERE _aqe_rn <= CEIL(_aqe_gsz * 0.1)"
        and the CTE itself is prepended separately in step 6.
        """
        for table_node in ast.find_all(exp.Table):
            if table_node.name == original_table:
                # Parse the replacement fragment and swap it in
                replacement = sqlglot.parse_one(
                    f"SELECT * FROM {fragment.table_ref}",
                    dialect=self.dialect,
                ).find(exp.From)

                if replacement:
                    # Replace just the FROM expression's table, not the whole FROM
                    from_node = table_node.parent
                    if from_node:
                        table_node.replace(
                            sqlglot.parse_one(
                                fragment.table_ref.split(" ")[0],
                                into=exp.Table,
                            )
                        )
                break

        return ast

    def _scale_aggregates(self, ast: exp.Expression, rate: float) -> exp.Expression:
        """
        Walks the AST and wraps SCALE_UP_AGGS with ROUND(agg / rate).

        Before: COUNT(*)
        After:  ROUND(COUNT(*) / 0.1)

        Before: SUM(revenue)
        After:  ROUND(SUM(revenue) / 0.1)

        AVG, MIN, MAX, MEDIAN are left completely untouched.
        """
        if rate <= 0 or rate >= 1.0:
            return ast   # no scaling needed for exact queries

        inverse = round(1.0 / rate, 6)

        for node in ast.find_all(exp.Count, exp.Sum, exp.Anonymous):
            func_name = (
                node.name.upper()
                if hasattr(node, "name")
                else type(node).__name__.upper()
            )

            if func_name not in SCALE_UP_AGGS:
                continue

            # Don't double-wrap if already scaled (idempotency guard)
            if isinstance(node.parent, exp.Round):
                continue

            scaled = exp.Round(
                this=exp.Mul(
                    this=node.copy(),
                    expression=exp.Literal.number(inverse),
                )
            )
            node.replace(scaled)

        return ast

    def _prepend_cte(self, ast: exp.Expression, cte_sql: str) -> exp.Expression:
        """
        Prepends a WITH clause to the query AST.
        If the query already has CTEs, appends to the existing WITH block.

        Input cte_sql is just the CTE body (no leading WITH keyword):
            _aqe_strat AS (SELECT ...)

        We parse it as a full statement then extract the CTE node.
        """
        # Parse a dummy query to get a typed CTE node
        wrapper  = sqlglot.parse_one(
            f"WITH {cte_sql} SELECT 1",
            dialect=self.dialect,
        )
        new_cte  = wrapper.find(exp.CTE)

        if new_cte is None:
            return ast   # parse failed, return unchanged

        existing_with = ast.find(exp.With)

        if existing_with:
            # Append to existing WITH block
            existing_with.append("expressions", new_cte)
        else:
            # Wrap the whole query in a new WITH
            ast = exp.With(
                expressions=[new_cte],
                this=ast,
            )

        return ast