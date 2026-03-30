from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

from rewriter.sample_builder import SampleFragment
from rewriter.agg_scaler import AggScaler, ScaleResult


@dataclass
class RewriteResult:
    """
    Output of the string-based rewriter.

    sql                : final rewritten SQL string
    warnings           : any warnings from scaling (e.g. COUNT DISTINCT)
    sample_rate        : rate used for sampling (1.0 for exact)
    sampling_strategy  : strategy name from SampleFragment
    """

    sql: str
    warnings: List[str]
    sample_rate: float
    sampling_strategy: str


class StringRewriter:
    """
    Very simple string-based query rewriter.

    Assumptions (for now):
      - Single-table SELECT queries.
      - Query is of the form: SELECT <select_list> FROM <table> ...
      - Aggregates in the SELECT list have no commas inside their argument list.

    This is intentionally minimal so you can work before the real SQL parser / AST
    is wired in. Later, this logic can be replaced by AST-based rewriting while
    keeping the same external contract.
    """

    def __init__(self, scaler: AggScaler | None = None):
        self.scaler = scaler or AggScaler()

    # ── Public API ──────────────────────────────────────────────────────────

    def rewrite(
        self,
        original_sql: str,
        fragment: SampleFragment,
    ) -> RewriteResult:
        """
        Rewrites a simple SELECT query to:
          1) inject the sampling fragment into the FROM clause
          2) scale aggregates in the SELECT list using AggScaler
        """

        # 1) Split SELECT ... FROM ...
        # We only support one FROM and single table for now.
        parts = re.split(r"\bFROM\b", original_sql, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            # If the query is not in the expected shape, return it unchanged.
            return RewriteResult(
                sql=original_sql,
                warnings=["Query shape not supported by StringRewriter; returned unchanged."],
                sample_rate=fragment.sample_rate,
                sampling_strategy=fragment.strategy,
            )

        select_part, from_and_rest = parts[0], parts[1]

        # 2) Parse and scale SELECT list
        scaled_select, scale_warnings = self._scale_select_list(
            select_part, fragment.sample_rate
        )

        # 3) Inject sampling into FROM clause
        rewritten_from = self._rewrite_from(from_and_rest, fragment.table_ref)

        # 4) Attach CTE (if any)
        core_sql = f"{scaled_select} FROM{rewritten_from}"
        if fragment.cte_sql:
            final_sql = f"WITH {fragment.cte_sql}\n{core_sql}"
        else:
            final_sql = core_sql

        return RewriteResult(
            sql=final_sql,
            warnings=scale_warnings,
            sample_rate=fragment.sample_rate,
            sampling_strategy=fragment.strategy,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _scale_select_list(self, select_part: str, rate: float) -> tuple[str, List[str]]:
        """
        Takes the 'SELECT ...' prefix and scales any aggregates found
        in the comma-separated list.
        """
        warnings: List[str] = []

        # Strip the leading SELECT keyword.
        body = re.sub(r"^\s*SELECT\s", "", select_part, flags=re.IGNORECASE)

        # Naive split on commas is fine for our simple aggregate shapes.
        exprs = [e.strip() for e in body.split(",") if e.strip()]
        scaled_exprs: List[str] = []

        for expr in exprs:
            result: ScaleResult = self.scaler.scale(expr, rate)
            scaled_exprs.append(result.scaled_sql)
            warnings.extend(result.warnings)

        scaled_select = "SELECT " + ", ".join(scaled_exprs)
        return scaled_select, warnings

    def _rewrite_from(self, from_and_rest: str, new_table_ref: str) -> str:
        """
        Replaces the base table after FROM with the sample fragment's table_ref,
        preserving trailing clauses (WHERE, GROUP BY, etc.).

        Input:  ' orders WHERE region = 'APAC''
        Output: ' orders USING SAMPLE ... WHERE region = 'APAC''
        """
        rest = from_and_rest.lstrip()
        if not rest:
            return f" {new_table_ref}"

        # Split once on whitespace: first token = original table, rest = tail clauses.
        tokens = rest.split(None, 1)
        tail = f" {tokens[1]}" if len(tokens) > 1 else ""
        return f" {new_table_ref}{tail}"

