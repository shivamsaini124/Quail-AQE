from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, List, Tuple

from rewriter.sample_builder import SampleFragment


@dataclass(frozen=True)
class MetricCI:
    """
    Confidence interval metadata for a single metric.

    estimate      : point estimate (already scaled to full data if applicable)
    se            : standard error estimate
    ci_low/ci_high: normal-approx CI bounds
    rel_half_width: (ci_high - estimate) / max(|estimate|, eps)
    """

    name: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    rel_half_width: float


@dataclass(frozen=True)
class ConfidenceReport:
    confidence_level: float
    metrics: List[MetricCI]
    warnings: List[str]


class ConfidenceEstimator:
    """
    Computes error/confidence for simple aggregate queries by running a companion
    "stats" query on the SAME sampled relation.

    Supported (string-based) aggregates in SELECT list:
      - COUNT(*)
      - SUM(<expr>)
      - AVG(<expr>)

    Notes:
      - For COUNT and SUM we use a Poisson/Bernoulli (Horvitz–Thompson) variance
        estimator assuming equal inclusion probability r:
          Var(T_hat) ≈ (1-r)/r^2 * sum_sample(y^2)
        where y=1 for COUNT(*) and y=<expr> for SUM(<expr>).
      - For AVG we use sample variance:
          SE(mean) ≈ sqrt(var_samp / n) * sqrt(1-r)
    """

    def build_stats_sql(self, original_sql: str, fragment: SampleFragment) -> Tuple[str, List[str]]:
        """
        Returns (stats_sql, metric_specs).

        metric_specs is a list of metric descriptors in the same order they appear
        in the original SELECT list; used to map DuckDB results back to metrics.
        """
        metric_specs, warnings = self._extract_metrics(original_sql)

        # Reuse the original query's FROM+tail, but swap the table for the sampled table_ref.
        from_tail = self._extract_from_tail(original_sql)
        from_tail = self._replace_from_table(from_tail, fragment.table_ref)

        # Build SELECT for required stats.
        select_cols: List[str] = []
        # Always include n = count of rows *after WHERE*, to support AVG SE.
        select_cols.append("COUNT(*) AS __aqe_n")

        for idx, spec in enumerate(metric_specs):
            kind = spec["kind"]
            expr = spec.get("expr")
            if kind == "count":
                select_cols.append(f"COUNT(*) AS __aqe_count_{idx}")
            elif kind == "sum":
                # Need sum and sumsq for variance estimator.
                select_cols.append(f"SUM({expr}) AS __aqe_sum_{idx}")
                select_cols.append(f"SUM(({expr}) * ({expr})) AS __aqe_sumsq_{idx}")
            elif kind == "avg":
                select_cols.append(f"AVG({expr}) AS __aqe_avg_{idx}")
                select_cols.append(f"VAR_SAMP({expr}) AS __aqe_var_{idx}")
            else:
                warnings.append(f"Unsupported metric kind: {kind!r}")

        core = f"SELECT {', '.join(select_cols)} FROM {from_tail}"
        if fragment.cte_sql:
            return f"WITH {fragment.cte_sql}\n{core}", warnings
        return core, warnings

    def compute(
        self,
        *,
        original_sql: str,
        fragment: SampleFragment,
        approx_result_row: Tuple[Any, ...],
        stats_row: Dict[str, Any],
        confidence_level: float = 0.95,
    ) -> ConfidenceReport:
        metric_specs, warnings = self._extract_metrics(original_sql)

        z = self._z_value(confidence_level)
        r = fragment.sample_rate
        eps = 1e-12

        metrics: List[MetricCI] = []
        n = float(stats_row.get("__aqe_n", 0) or 0)

        for i, spec in enumerate(metric_specs):
            kind = spec["kind"]
            name = spec["raw"]

            if kind == "unknown":
                warnings.append(
                    f"No CI for non-aggregate SELECT item {name!r} (e.g. GROUP BY key)."
                )
                continue

            estimate = float(approx_result_row[i])

            if kind == "count":
                count_s = float(stats_row.get(f"__aqe_count_{i}", 0) or 0)
                var = (1.0 - r) / (r * r) * count_s
                se = math.sqrt(max(var, 0.0))

            elif kind == "sum":
                sumsq = float(stats_row.get(f"__aqe_sumsq_{i}", 0) or 0)
                var = (1.0 - r) / (r * r) * sumsq
                se = math.sqrt(max(var, 0.0))

            elif kind == "avg":
                var_samp = float(stats_row.get(f"__aqe_var_{i}", 0) or 0)
                if n <= 1:
                    se = float("inf")
                    warnings.append(f"Not enough rows to estimate SE for AVG in metric {name!r}.")
                else:
                    se = math.sqrt(max(var_samp, 0.0) / n) * math.sqrt(max(1.0 - r, 0.0))
            else:
                se = float("nan")
                warnings.append(f"No CI computed for unsupported metric {name!r}.")

            half = z * se
            ci_low = estimate - half
            ci_high = estimate + half
            rel_half = half / max(abs(estimate), eps)

            metrics.append(
                MetricCI(
                    name=name,
                    estimate=estimate,
                    se=se,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    rel_half_width=rel_half,
                )
            )

        return ConfidenceReport(
            confidence_level=confidence_level,
            metrics=metrics,
            warnings=warnings,
        )

    # ── Parsing helpers ───────────────────────────────────────────────────

    def _extract_from_tail(self, sql: str) -> str:
        parts = re.split(r"\bFROM\b", sql, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError("Query must contain FROM for confidence estimation.")
        return parts[1].strip()

    def _replace_from_table(self, from_tail: str, new_table_ref: str) -> str:
        tokens = from_tail.split(None, 1)
        tail = f" {tokens[1]}" if len(tokens) > 1 else ""
        return f"{new_table_ref}{tail}"

    def _extract_metrics(self, sql: str) -> Tuple[List[Dict[str, str]], List[str]]:
        warnings: List[str] = []
        select_body = re.split(r"\bFROM\b", sql, maxsplit=1, flags=re.IGNORECASE)[0]
        select_body = re.sub(r"^\s*SELECT\s", "", select_body, flags=re.IGNORECASE).strip()

        # Same limitations as StringRewriter: no commas inside args.
        exprs = [e.strip() for e in select_body.split(",") if e.strip()]
        specs: List[Dict[str, str]] = []

        for expr in exprs:
            norm = expr.strip()
            upper = norm.upper()
            if upper.startswith("COUNT(") and "DISTINCT" not in upper:
                specs.append({"kind": "count", "raw": norm})
            elif upper.startswith("SUM("):
                inner = self._inner_arg(norm)
                specs.append({"kind": "sum", "expr": inner, "raw": norm})
            elif upper.startswith("AVG("):
                inner = self._inner_arg(norm)
                specs.append({"kind": "avg", "expr": inner, "raw": norm})
            else:
                warnings.append(f"Unsupported SELECT expression for CI: {norm!r}")
                specs.append({"kind": "unknown", "raw": norm})

        return specs, warnings

    def _inner_arg(self, call: str) -> str:
        m = re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*\((.*)\)\s*$", call)
        if not m:
            return "*"
        return m.group(1).strip()

    # ── Stats helpers ──────────────────────────────────────────────────────

    def _z_value(self, confidence_level: float) -> float:
        # Minimal set; avoids extra deps (scipy).
        # 90%, 95%, 99%
        if abs(confidence_level - 0.90) < 1e-9:
            return 1.6448536269514722
        if abs(confidence_level - 0.95) < 1e-9:
            return 1.959963984540054
        if abs(confidence_level - 0.99) < 1e-9:
            return 2.5758293035489004
        # Fallback: default to 95%
        return 1.959963984540054

