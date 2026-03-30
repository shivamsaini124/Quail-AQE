from rewriter.sample_builder import SampleFragment
from rewriter.rewriter import StringRewriter


rewriter = StringRewriter()


def test_uniform_injection_and_scaling():
    original = "SELECT COUNT(*), AVG(price) FROM orders WHERE region = 'APAC'"
    fragment = SampleFragment(
        cte_sql=None,
        table_ref="orders USING SAMPLE 10.0% (bernoulli)",
        sample_rate=0.1,
        strategy="uniform",
    )

    result = rewriter.rewrite(original, fragment)

    # Sampling injected
    assert "USING SAMPLE 10.0% (bernoulli)" in result.sql
    # COUNT scaled, AVG not scaled
    assert "ROUND(COUNT(*) * 10.0)" in result.sql
    assert "AVG(price)" in result.sql
    print("✅ test_uniform_injection_and_scaling passed")
    print("   SQL:", result.sql)


def test_stratified_injection_with_cte():
    original = "SELECT SUM(revenue) FROM orders GROUP BY region"
    fragment = SampleFragment(
        cte_sql=(
            "_aqe_strat AS (\n"
            "    SELECT *,\n"
            "        ROW_NUMBER() OVER (\n"
            "            PARTITION BY region ORDER BY abs(hash(id))\n"
            "        ) AS _aqe_rn,\n"
            "        COUNT(*) OVER (\n"
            "            PARTITION BY region\n"
            "        ) AS _aqe_gsz\n"
            "    FROM orders\n"
            ")"
        ),
        table_ref="_aqe_strat WHERE _aqe_rn <= CEIL(_aqe_gsz * 0.1)",
        sample_rate=0.1,
        strategy="stratified",
    )

    result = rewriter.rewrite(original, fragment)

    # CTE attached
    assert result.sql.strip().startswith("WITH _aqe_strat AS")
    # SUM scaled
    assert "ROUND(SUM(revenue) * 10.0)" in result.sql
    print("✅ test_stratified_injection_with_cte passed")
    print("   SQL:", result.sql)


def test_unsupported_shape_returns_original():
    original = "INSERT INTO orders VALUES (1, 2, 3)"
    fragment = SampleFragment(
        cte_sql=None,
        table_ref="orders USING SAMPLE 10.0% (bernoulli)",
        sample_rate=0.1,
        strategy="uniform",
    )

    result = rewriter.rewrite(original, fragment)
    assert result.sql == original
    assert any("not supported" in w.lower() for w in result.warnings)
    print("✅ test_unsupported_shape_returns_original passed")


if __name__ == "__main__":
    test_uniform_injection_and_scaling()
    test_stratified_injection_with_cte()
    test_unsupported_shape_returns_original()

