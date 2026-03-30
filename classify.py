from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from features import Phase1Features, Phase2Features
from parser import QueryIR

Strategy = Literal["EXACT", "SAMPLING", "FILTERED_SAMPLING", "HYBRID"]


@dataclass(frozen=True)
class SamplingPlan:
    """
    Output contract of the classifier/planner for the rewriter layer.

    strategy        : which approximation strategy to use
    sample_rate     : fraction in (0, 1], e.g. 0.1 for 10%
    stratify_on     : optional list of columns to stratify on (e.g. GROUP BY keys)
    table_row_count : optional row count estimate for sample-size gating
    hash_key        : stable key column used for deterministic hash sampling/order
    reason          : human-readable reason label from the classifier
    """

    strategy: Strategy
    sample_rate: float
    stratify_on: list[str] | None = None
    table_row_count: int | None = None
    hash_key: str | None = "id"
    reason: str = ""


def make_sampling_plan(
    p1: Phase1Features,
    p2: Phase2Features,
    ir: QueryIR,
    *,
    hash_key: str | None = "id",
) -> SamplingPlan:
    """
    Convenience wrapper: choose strategy + derive a plan the rewriter can use.
    """
    strategy, reason = choose_strategy(p1, p2, ir)

    stratify_on = ir.groupby_cols if ir.groupby_cols else None
    sample_rate = float(p2.sample_size)

    # Clamp to sane bounds for safety.
    if sample_rate <= 0.0:
        sample_rate = 0.01
    if sample_rate > 1.0:
        sample_rate = 1.0

    return SamplingPlan(
        strategy=strategy,
        sample_rate=sample_rate,
        stratify_on=stratify_on,
        table_row_count=p2.table_row_count,
        hash_key=hash_key,
        reason=reason,
    )


def choose_strategy(
    p1: Phase1Features,
    p2: Phase2Features,
    ir: QueryIR
) -> tuple[Strategy, str]:

 
    if p2.selectivity < 0.01:
        return "EXACT", "very_low_selectivity"

    
   
    if p1.num_joins >= 2:
        return "HYBRID", "many_joins"

    
    if p2.variance > 1e5:
        return "HYBRID", "high_variance"

    
    if p1.has_groupby:
        if p2.group_count > 1000 or p2.group_count <= 5:
            return "HYBRID", "extreme_group_count"

    
    if p2.sample_size < 0.1:
        return "HYBRID", "small_sample_size"

   
    has_filters = p1.num_filters >= 1
    has_range = any(p["is_range"] for p in ir.predicates)

    if (
        has_filters
        and has_range
        and p2.selectivity >= 0.1
        and p2.variance < 1e5
    ):
        return "FILTERED_SAMPLING", "range_filter_sampling"

   
    if (
        p1.num_filters == 0
        and p1.num_joins <= 1
        and not p1.has_groupby
        and p2.sample_size > 0.1
        and p2.variance < 1e5
    ):
        return "SAMPLING", "simple_sampling"

    return "SAMPLING", "fallback"