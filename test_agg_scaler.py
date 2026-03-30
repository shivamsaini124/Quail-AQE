# test_agg_scaler.py
from rewriter.agg_scaler import AggScaler

scaler = AggScaler()

# ── Test 1: COUNT gets scaled ─────────────────────────────────────
def test_count_scaled():
    result = scaler.scale("COUNT(*)", rate=0.1)
    assert result.was_scaled is True
    assert "ROUND" in result.scaled_sql
    assert "10.0" in result.scaled_sql      # 1/0.1 = 10
    print("✅ test_count_scaled:", result.scaled_sql)

# ── Test 2: SUM gets scaled ───────────────────────────────────────
def test_sum_scaled():
    result = scaler.scale("SUM(revenue)", rate=0.2)
    assert result.was_scaled is True
    assert "ROUND" in result.scaled_sql
    assert "5.0" in result.scaled_sql       # 1/0.2 = 5
    print("✅ test_sum_scaled:", result.scaled_sql)

# ── Test 3: AVG not scaled ────────────────────────────────────────
def test_avg_not_scaled():
    result = scaler.scale("AVG(price)", rate=0.1)
    assert result.was_scaled is False
    assert result.scaled_sql == "AVG(price)"   # unchanged
    print("✅ test_avg_not_scaled:", result.scaled_sql)

# ── Test 4: MIN not scaled ────────────────────────────────────────
def test_min_not_scaled():
    result = scaler.scale("MIN(price)", rate=0.1)
    assert result.was_scaled is False
    assert result.scaled_sql == "MIN(price)"
    print("✅ test_min_not_scaled:", result.scaled_sql)

# ── Test 5: COUNT DISTINCT not scaled but warns ───────────────────
def test_count_distinct_warns():
    result = scaler.scale("COUNT(DISTINCT user_id)", rate=0.1)
    assert result.was_scaled is False          # not scaled
    assert len(result.warnings) > 0            # but warns about low rate
    print("✅ test_count_distinct_warns:", result.warnings[0])

# ── Test 6: COUNT DISTINCT at safe rate — no warning ─────────────
def test_count_distinct_safe_rate():
    result = scaler.scale("COUNT(DISTINCT user_id)", rate=0.5)
    assert result.was_scaled is False
    assert len(result.warnings) == 0           # no warning at 50%
    print("✅ test_count_distinct_safe_rate — no warnings")

# ── Test 7: scale_many handles multiple aggregates ────────────────
def test_scale_many():
    exprs = ["COUNT(*)", "SUM(revenue)", "AVG(price)"]
    results = scaler.scale_many(exprs, rate=0.1)
    assert results[0].was_scaled is True    # COUNT scaled
    assert results[1].was_scaled is True    # SUM scaled
    assert results[2].was_scaled is False   # AVG not scaled
    print("✅ test_scale_many passed")

# ── Test 8: case insensitive ──────────────────────────────────────
def test_lowercase_agg():
    result = scaler.scale("count(*)", rate=0.1)
    assert result.was_scaled is True
    print("✅ test_lowercase_agg:", result.scaled_sql)


if __name__ == "__main__":
    test_count_scaled()
    test_sum_scaled()
    test_avg_not_scaled()
    test_min_not_scaled()
    test_count_distinct_warns()
    test_count_distinct_safe_rate()
    test_scale_many()
    test_lowercase_agg()