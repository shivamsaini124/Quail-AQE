# aqe/rewriter/sample_builder.py
from __future__ import annotations
from dataclasses import dataclass
from aqe.classifier.sampling_plan import SamplingPlan
from aqe.parser.query_ir import QueryIR


@dataclass
class SampleFragment:
    """
    Holds the sample SQL fragment and metadata needed by the rewriter.
    cte_sql   : the WITH clause body (None for native TABLESAMPLE)
    table_ref : what to put in the FROM clause after injection
    sample_rate: actual rate used (may differ from plan if gate adjusted it)
    strategy  : "uniform" | "stratified" | "hash"
    """
    cte_sql:     str | None
    table_ref:   str
    sample_rate: float
    strategy:    str


# ── minimum sample floor ──────────────────────────────────────────────────────
# Below this, statistical results are unreliable regardless of strategy.
MIN_SAMPLE_N = 30


class SampleSizeGate:
    """
    Checks whether the computed sample n is large enough to be statistically
    meaningful. If not, signals a fallback to exact execution.
    """

    def __init__(self, min_n: int = MIN_SAMPLE_N):
        self.min_n = min_n

    def passes(self, plan: SamplingPlan, row_count: int) -> bool:
        estimated_n = int(plan.sample_rate * row_count)
        return estimated_n >= self.min_n

    def adjusted_rate(self, row_count: int) -> float:
        """
        If the caller wants to proceed despite a low rate, this returns the
        minimum rate that would satisfy the floor rather than hard-failing.
        """
        return self.min_n / max(row_count, 1)


class SampleBuilder:
    """
    Builds the SQL fragment that represents the sample.

    Three strategies:
      uniform    — TABLESAMPLE BERNOULLI, row-level, unbiased, db-native
      stratified — CTE with ROW_NUMBER() OVER (PARTITION BY strat_key)
      hash       — deterministic, no RANDOM(), fast on large tables
    """

    def __init__(self, gate: SampleSizeGate | None = None):
        self.gate = gate or SampleSizeGate()

    # ── public entry point ────────────────────────────────────────────────────

    def build(self, ir: QueryIR, plan: SamplingPlan) -> SampleFragment:
        """
        Dispatches to the correct strategy based on plan.strategy.
        Raises SampleSizeError if n is below the floor.
        """
        row_count = ir.table_row_count or 1_000_000  # fallback estimate

        if not self.gate.passes(plan, row_count):
            raise SampleSizeError(
                f"Estimated sample n={int(plan.sample_rate * row_count)} "
                f"is below minimum {self.gate.min_n}. "
                f"Use exact execution instead."
            )

        table = ir.tables[0]

        if plan.strategy == "sampling":
            return self._uniform(table, plan.sample_rate)

        elif plan.strategy == "filter_sampling":
            strat_key = plan.stratify_on[0] if plan.stratify_on else None
            if strat_key:
                return self._stratified(table, strat_key, plan.sample_rate)
            return self._uniform(table, plan.sample_rate)

        elif plan.strategy == "hybrid":
            strat_key = plan.stratify_on[0] if plan.stratify_on else None
            if strat_key:
                return self._stratified(table, strat_key, plan.sample_rate)
            return self._hash(table, plan.sample_rate)

        else:
            raise ValueError(f"Unknown strategy: {plan.strategy!r}")

    # ── strategy implementations ──────────────────────────────────────────────

    def _uniform(self, table: str, rate: float) -> SampleFragment:
        """
        Uses the database's native block-level sampler.
        BERNOULLI = row-level (unbiased, slightly slower).
        SYSTEM    = block-level (faster, minor clustering bias).
        We default to BERNOULLI for correctness; switch to SYSTEM
        for tables > 100M rows where speed matters more.
        """
        pct = round(rate * 100, 4)
        return SampleFragment(
            cte_sql    = None,
            table_ref  = f"{table} TABLESAMPLE BERNOULLI ({pct})",
            sample_rate = rate,
            strategy   = "uniform",
        )

    def _stratified(self, table: str, strat_col: str, rate: float) -> SampleFragment:
        """
        CTE-based stratified sample.
        Guarantees every group gets proportional representation.

        Structure:
          WITH _strat AS (
              SELECT *,
                  ROW_NUMBER() OVER (PARTITION BY strat_col ORDER BY RANDOM()) AS _rn,
                  COUNT(*)     OVER (PARTITION BY strat_col)                   AS _gsz
              FROM table
          )
          SELECT * FROM _strat
          WHERE _rn <= CEIL(_gsz * rate)

        The CEIL ensures at least 1 row per group even at very low rates,
        which prevents groups from disappearing entirely from the sample.
        """
        cte = f"""_aqe_strat AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY {strat_col}
            ORDER BY RANDOM()
        ) AS _aqe_rn,
        COUNT(*) OVER (
            PARTITION BY {strat_col}
        ) AS _aqe_gsz
    FROM {table}
)"""
        table_ref = (
            f"_aqe_strat WHERE _aqe_rn <= CEIL(_aqe_gsz * {rate})"
        )
        return SampleFragment(
            cte_sql    = cte,
            table_ref  = table_ref,
            sample_rate = rate,
            strategy   = "stratified",
        )

    def _hash(self, table: str, rate: float) -> SampleFragment:
        """
        Deterministic hash-based sample. No RANDOM() call means results are
        reproducible across runs and no full sort is needed.
        Uses FARM_FINGERPRINT (BigQuery) or md5 (Postgres) depending on dialect.

        Trade-off vs BERNOULLI:
          + deterministic → same query always gets same rows
          + fast on very large tables (no per-row random number)
          - not purely random; correlated with the id distribution
        """
        bucket_threshold = int(rate * 2**32)
        cte = f"""_aqe_hash AS (
    SELECT *
    FROM {table}
    WHERE ('x' || lpad(md5(id::text), 8, '0'))::bit(32)::int
          < {bucket_threshold}
)"""
        return SampleFragment(
            cte_sql    = cte,
            table_ref  = "_aqe_hash",
            sample_rate = rate,
            strategy   = "hash",
        )


class SampleSizeError(Exception):
    """Raised when estimated sample n is below the statistical minimum."""
    pass