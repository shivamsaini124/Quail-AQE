# rewriter/sample_builder.py
from __future__ import annotations
from dataclasses import dataclass


# ─────────────────────────────────────────────
# MOCK CLASSES - replace with real imports later
# from parser import QueryIR
# from classifier import SamplingPlan
# ─────────────────────────────────────────────

@dataclass
class QueryIR:
    """Temporary mock — will be replaced by real parser.QueryIR"""
    tables: list[str]
    table_row_count: int | None = None
    stratify_on: list[str] | None = None
    # Optional stable key (e.g., primary key) used for deterministic hashing/order.
    # Keep this name flexible so it can map to your real IR later.
    sample_key: str | None = "id"

@dataclass
class SamplingPlan:
    """Temporary mock — will be replaced by real classifier.SamplingPlan"""
    strategy: str          # "sampling" | "filter_sampling" | "hybrid"
    sample_rate: float
    stratify_on: list[str] | None = None
    predicted_error: float = 0.05
    # Optional override for the key used in hash-based sampling/order.
    hash_key: str | None = None


# ─────────────────────────────────────────────
# REAL CODE STARTS HERE
# ─────────────────────────────────────────────

MIN_SAMPLE_N = 30  # below this, results are statistically unreliable


class SampleSizeError(Exception):
    """Raised when estimated sample n is below the statistical minimum."""
    pass


@dataclass
class SampleFragment:
    """
    The output of SampleBuilder.
    Carries everything the rewriter needs to inject the sample into the query.
    
    cte_sql     : the WITH clause body. None for TABLESAMPLE (uniform).
    table_ref   : what goes in the FROM clause after injection.
    sample_rate : actual rate used (may differ from plan if adjusted).
    strategy    : which method was used.
    """
    cte_sql: str | None
    table_ref: str
    sample_rate: float
    strategy: str


class SampleSizeGate:
    """
    Checks whether the planned sample is large enough to be statistically valid.
    
    Why 30? Central Limit Theorem - below 30 samples, estimates become 
    unreliable regardless of which sampling method you use.
    """

    def __init__(self, min_n: int = MIN_SAMPLE_N):
        self.min_n = min_n

    def passes(self, plan: SamplingPlan, row_count: int) -> bool:
        estimated_n = int(plan.sample_rate * row_count)
        return estimated_n >= self.min_n

    def adjusted_rate(self, row_count: int) -> float:
        """Returns the minimum rate that would give us exactly min_n rows."""
        return self.min_n / max(row_count, 1)


class SampleBuilder:
    """
    Builds the SQL fragment that represents the sample.

    3 strategies:
      uniform    — DuckDB USING SAMPLE, db-native, fast, unbiased
      stratified — CTE with ROW_NUMBER() OVER (PARTITION BY group_col), ordered by hash(key)
      hash       — deterministic hash(key) predicate, reproducible
    """

    def __init__(self, gate: SampleSizeGate | None = None):
        self.gate = gate or SampleSizeGate()

    def build(self, ir: QueryIR, plan: SamplingPlan) -> SampleFragment:
        row_count = ir.table_row_count or 1_000_000  # fallback estimate

        # Gate check — is n large enough?
        if not self.gate.passes(plan, row_count):
            raise SampleSizeError(
                f"Estimated n={int(plan.sample_rate * row_count)} "
                f"is below minimum {self.gate.min_n}. "
                f"Fall back to exact execution."
            )

        table = ir.tables[0]
        hash_key = plan.hash_key or ir.sample_key or "id"

        if plan.strategy == "sampling":
            return self._uniform(table, plan.sample_rate)

        elif plan.strategy == "filter_sampling":
            strat_key = plan.stratify_on[0] if plan.stratify_on else None
            if strat_key:
                return self._stratified(table, strat_key, plan.sample_rate, hash_key=hash_key)
            return self._uniform(table, plan.sample_rate)

        elif plan.strategy == "hybrid":
            strat_key = plan.stratify_on[0] if plan.stratify_on else None
            if strat_key:
                return self._stratified(table, strat_key, plan.sample_rate, hash_key=hash_key)
            return self._hash(table, plan.sample_rate, hash_key=hash_key)

        else:
            raise ValueError(f"Unknown strategy: {plan.strategy!r}")

    # ── 3 private methods, one per strategy ──────────────────────────

    def _uniform(self, table: str, rate: float) -> SampleFragment:
        """
        Uses DuckDB's native sampling.
        No CTE needed — just modifies the table reference in the FROM clause.

        Example output table_ref:
            orders USING SAMPLE 10% (bernoulli)
        """
        pct = round(rate * 100, 4)
        return SampleFragment(
            cte_sql=None,
            table_ref=f"{table} USING SAMPLE {pct}% (bernoulli)",
            sample_rate=rate,
            strategy="uniform",
        )

    def _stratified(self, table: str, strat_col: str, rate: float, *, hash_key: str) -> SampleFragment:
        """
        CTE-based stratified sample.
        Guarantees every group gets proportional rows — none disappear.
        
        CEIL ensures at least 1 row per group even at very low rates.
        
        Example: 3 regions, 10% rate → each region loses exactly 90% of rows.
        """
        # Optimization: avoid ORDER BY random() (expensive). Ordering by a stable hash
        # gives a pseudo-random, reproducible permutation within each group.
        cte = (
            f"_aqe_strat AS (\n"
            f"    SELECT *,\n"
            f"        ROW_NUMBER() OVER (\n"
            f"            PARTITION BY {strat_col} ORDER BY abs(hash({hash_key}))\n"
            f"        ) AS _aqe_rn,\n"
            f"        COUNT(*) OVER (\n"
            f"            PARTITION BY {strat_col}\n"
            f"        ) AS _aqe_gsz\n"
            f"    FROM {table}\n"
            f")"
        )
        return SampleFragment(
            cte_sql=cte,
            table_ref=f"_aqe_strat WHERE _aqe_rn <= CEIL(_aqe_gsz * {rate})",
            sample_rate=rate,
            strategy="stratified",
        )

    def _hash(self, table: str, rate: float, *, hash_key: str) -> SampleFragment:
        """
        Deterministic hash-based sample.
        No random() = same query always returns the same rows.

        DuckDB-friendly approach: take abs(hash(key)) modulo a big number M,
        then keep rows below floor(rate * M).
        """
        modulus = 1_000_000
        bucket_threshold = int(rate * modulus)
        cte = (
            f"_aqe_hash AS (\n"
            f"    SELECT * FROM {table}\n"
            f"    WHERE (abs(hash({hash_key})) % {modulus}) < {bucket_threshold}\n"
            f")"
        )
        return SampleFragment(
            cte_sql=cte,
            table_ref="_aqe_hash",
            sample_rate=rate,
            strategy="hash",
        )