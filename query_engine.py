from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb

from bypass import should_approximate
from parser import parse, QueryIR
from classify import SamplingPlan
from rewriter.sample_builder import SampleBuilder, SampleFragment, SampleSizeError
from rewriter.rewriter import StringRewriter, RewriteResult
from confidence import ConfidenceEstimator, ConfidenceReport


@dataclass(frozen=True)
class QueryResponse:
    original_sql: str
    executed_sql: str
    is_approx: bool
    strategy: str
    sample_rate: float
    rows: List[Tuple[Any, ...]]
    columns: List[str]
    warnings: List[str]
    confidence: ConfidenceReport | None


class QueryEngine:
    """
    Minimal AQE query runner for DuckDB.

    - Parses SQL to IR
    - Applies bypass rules
    - If approximate: uses provided SamplingPlan + SampleBuilder + StringRewriter
    - Executes final SQL in DuckDB
    - Computes confidence intervals using a companion stats query
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection | None = None,
        *,
        builder: SampleBuilder | None = None,
        rewriter: StringRewriter | None = None,
        estimator: ConfidenceEstimator | None = None,
    ):
        self.con = con or duckdb.connect(database=":memory:")
        self.builder = builder or SampleBuilder()
        self.rewriter = rewriter or StringRewriter()
        self.estimator = estimator or ConfidenceEstimator()

    def execute_exact(self, sql: str) -> QueryResponse:
        cur = self.con.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return QueryResponse(
            original_sql=sql,
            executed_sql=sql,
            is_approx=False,
            strategy="EXACT",
            sample_rate=1.0,
            rows=rows,
            columns=cols,
            warnings=[],
            confidence=None,
        )

    def execute_approx(
        self,
        sql: str,
        *,
        plan: SamplingPlan,
        confidence_level: float = 0.95,
    ) -> QueryResponse:
        ir: QueryIR = parse(sql)
        ok, reason = should_approximate(ir)
        if not ok or plan.strategy == "EXACT":
            resp = self.execute_exact(sql)
            return replace(resp, warnings=[reason])

        try:
            fragment: SampleFragment = self.builder.build(ir, plan)
        except SampleSizeError as e:
            resp = self.execute_exact(sql)
            return replace(resp, warnings=[str(e)])

        rewrite: RewriteResult = self.rewriter.rewrite(sql, fragment)

        # Execute rewritten SQL
        cur = self.con.execute(rewrite.sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []

        # Confidence: compute stats query using same fragment (same sample)
        conf: ConfidenceReport | None = None
        warnings: List[str] = list(rewrite.warnings)
        if rows:
            stats_sql, ci_warnings = self.estimator.build_stats_sql(sql, fragment)
            warnings.extend(ci_warnings)
            stats_cur = self.con.execute(stats_sql)
            stats_row_tuple = stats_cur.fetchone()
            stats_cols = [d[0] for d in stats_cur.description] if stats_cur.description else []
            stats_row: Dict[str, Any] = dict(zip(stats_cols, stats_row_tuple)) if stats_row_tuple else {}

            conf = self.estimator.compute(
                original_sql=sql,
                fragment=fragment,
                approx_result_row=rows[0],
                stats_row=stats_row,
                confidence_level=confidence_level,
            )

        return QueryResponse(
            original_sql=sql,
            executed_sql=rewrite.sql,
            is_approx=True,
            strategy=plan.strategy,
            sample_rate=fragment.sample_rate,
            rows=rows,
            columns=cols,
            warnings=warnings,
            confidence=conf,
        )

