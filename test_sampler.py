# test_sampler.py
from parser import parse
from classify import SamplingPlan
from rewriter.sample_builder import (
    SampleBuilder,
    SampleSizeError,
)

builder = SampleBuilder()

# ── Test 1: Uniform strategy ──────────────────────────────────────
def test_uniform():
    ir = parse("SELECT COUNT(*) FROM orders")
    plan = SamplingPlan(strategy="SAMPLING", sample_rate=0.1, table_row_count=100_000)
    frag = builder.build(ir, plan)

    assert frag.strategy == "uniform"
    assert frag.cte_sql is None                        # no CTE for uniform
    assert "USING SAMPLE" in frag.table_ref
    assert "bernoulli" in frag.table_ref.lower()
    assert "10.0" in frag.table_ref
    print("✅ test_uniform passed:", frag.table_ref)

# ── Test 2: Stratified strategy ───────────────────────────────────
def test_stratified():
    ir = parse("SELECT region, COUNT(*) FROM orders GROUP BY region")
    plan = SamplingPlan(
        strategy="FILTERED_SAMPLING",
        sample_rate=0.1,
        stratify_on=["region"],
        table_row_count=100_000,
        hash_key="id",
    )
    frag = builder.build(ir, plan)

    assert frag.strategy == "stratified"
    assert frag.cte_sql is not None
    assert "ROW_NUMBER() OVER" in frag.cte_sql
    assert "PARTITION BY region" in frag.cte_sql
    assert "hash" in frag.cte_sql.lower()
    assert "CEIL" in frag.table_ref
    print("✅ test_stratified passed")
    print("   CTE:", frag.cte_sql)

# ── Test 3: Hash strategy ─────────────────────────────────────────
def test_hash():
    ir = parse("SELECT COUNT(*) FROM orders WHERE a = 1")
    plan = SamplingPlan(strategy="HYBRID", sample_rate=0.1, table_row_count=100_000, hash_key="id")
    frag = builder.build(ir, plan)

    assert frag.strategy == "hash"
    assert "hash" in frag.cte_sql.lower()
    assert frag.table_ref == "_aqe_hash"
    print("✅ test_hash passed")

# ── Test 4: Gate blocks tiny samples ─────────────────────────────
def test_gate_blocks_small_n():
    ir = parse("SELECT COUNT(*) FROM orders")
    plan = SamplingPlan(strategy="SAMPLING", sample_rate=0.001, table_row_count=100)  # n ~ 0
    try:
        builder.build(ir, plan)
        print("❌ test_gate_blocks_small_n FAILED — should have raised")
    except SampleSizeError as e:
        print("✅ test_gate_blocks_small_n passed:", e)

# ── Test 5: Fallback to uniform when no stratify_on ──────────────
def test_filter_sampling_no_strat_key():
    ir = parse("SELECT COUNT(*) FROM orders")
    plan = SamplingPlan(strategy="FILTERED_SAMPLING", sample_rate=0.1, stratify_on=None, table_row_count=100_000)
    frag = builder.build(ir, plan)

    assert frag.strategy == "uniform"   # falls back to uniform
    print("✅ test_filter_sampling_no_strat_key passed")


if __name__ == "__main__":
    test_uniform()
    test_stratified()
    test_hash()
    test_gate_blocks_small_n()
    test_filter_sampling_no_strat_key()