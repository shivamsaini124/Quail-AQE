from parser import parse
from bypass import should_approximate

test_queries = [
    # (label, sql, expect_approximate)
    ("Simple SELECT with COUNT",
     "SELECT COUNT(*) FROM orders",
     True),

    ("INSERT — should bypass",
     "INSERT INTO orders VALUES (1, 'test')",
     False),

    ("DISTINCT — should bypass",
     "SELECT DISTINCT customer_id FROM orders",
     False),

    ("No aggregation — should bypass",
     "SELECT * FROM orders WHERE id = 1",
     False),

    ("Subquery + many tables — should bypass",
     "SELECT COUNT(*) FROM a JOIN b ON a.id=b.id JOIN c ON b.id=c.id JOIN d ON c.id=d.id WHERE a.x = (SELECT MAX(x) FROM e)",
     False),

    ("GROUP BY with SUM",
     "SELECT region, SUM(revenue) FROM sales GROUP BY region",
     True),
]

print(f"{'Test':<40} {'Approve?':<10} {'Reason':<20} {'Pass?'}")
print("-" * 80)
for label, sql, expected in test_queries:
    ir = parse(sql)
    approve, reason = should_approximate(ir)
    passed = "✓" if approve == expected else "✗ FAIL"
    print(f"{label:<40} {str(approve):<10} {reason:<20} {passed}")

import json

print("\n\nParser Output Inspection")
print("=" * 60)

inspect_queries = [
    ("Simple COUNT", "SELECT COUNT(*) FROM orders"),
    ("WHERE filter", "SELECT COUNT(*) FROM orders WHERE status = 'open'"),
    ("JOIN + GROUP BY", "SELECT region, SUM(revenue) FROM sales JOIN customers ON sales.cid = customers.id GROUP BY region"),
    ("Subquery", "SELECT COUNT(*) FROM orders WHERE id IN (SELECT order_id FROM items)"),
]

for label, sql in inspect_queries:
    ir = parse(sql)
    approve, reason = should_approximate(ir)
    print(f"\n--- {label} ---")
    print(f"  SQL          : {sql}")
    print(f"  query_type   : {ir.query_type}")
    print(f"  tables       : {ir.tables}")
    print(f"  predicates   : {ir.predicates}")
    print(f"  joins        : {ir.joins}")
    print(f"  groupby_cols : {ir.groupby_cols}")
    print(f"  aggregations : {ir.aggregations}")
    print(f"  has_subquery : {ir.has_subquery}")
    print(f"  has_order_by : {ir.has_order_by}")
    print(f"  has_distinct : {ir.has_distinct}")
    print(f"  bypass?      : approve={approve}, reason={reason}")