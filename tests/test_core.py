import json
from pathlib import Path

from text2sql import db
from text2sql.llm import extract_sql, messages


def test_db_and_readonly(tmp_path):
    p = tmp_path / "shop.db"
    db.build_db(p)
    assert db.run(p, "SELECT COUNT(*) FROM orders") == [(2000,)]
    assert db.run(p, "DROP TABLE orders") is None
    assert db.run(p, "SELECT 1; DROP TABLE orders") is None
    assert db.run(p, "SELECT COUNT(*) FROM orders") == [(2000,)]  # still there


def test_same_ignores_order_and_float_noise():
    assert db.same([(1, 2.00001), (3, 4)], [(3, 4), (1, 2.0)])
    assert not db.same(None, [(1,)])


def test_extract_sql_and_messages():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1;"
    assert messages("q", (("a", "b"),))[-1] == {"role": "user", "content": "q"}


def test_seed_gold_sql_all_runs(tmp_path):
    p = tmp_path / "shop.db"
    db.build_db(p)
    for line in Path("data/seeds.jsonl").read_text().splitlines():
        s = json.loads(line)
        assert db.run(p, s["sql"]), s["id"]  # non-empty result, else dropped by data.py
