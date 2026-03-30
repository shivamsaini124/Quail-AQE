from __future__ import annotations
from typing import Literal

from features import Phase1Features, Phase2Features
from parser import QueryIR

Strategy = Literal["EXACT", "SAMPLING", "FILTERED_SAMPLING", "HYBRID"]


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