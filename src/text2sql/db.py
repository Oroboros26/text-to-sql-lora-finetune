"""Synthetic shop DB, read-only SQL execution, result comparison."""
import random
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

SCHEMA = (Path(__file__).parent / "schema.sql").read_text()
CATEGORIES = ["shoes", "jackets", "packs", "socks", "tents"]
COUNTRIES = ["FR", "US", "JP", "DE", "CA"]
STATUSES = ["shipped", "returned", "cancelled", "pending"]
ADJ = ["Trail", "Alpine", "Summit", "Urban"]


def build_db(path: str | Path, seed: int = 42) -> None:
    rng = random.Random(seed)
    Path(path).unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO stores VALUES (?,?,?)",
                    [(i + 1, f"Store {c}", c) for i, c in enumerate(COUNTRIES)])
    con.executemany("INSERT INTO products VALUES (?,?,?,?)",
                    [(i + 1, f"{rng.choice(ADJ)} {c.title()} {i + 1}", c, round(rng.uniform(10, 300), 2))
                     for i in range(40) for c in [rng.choice(CATEGORIES)]])
    start = date(2024, 1, 1)
    for oid in range(1, 2001):
        d = start + timedelta(days=rng.randrange(366))
        con.execute("INSERT INTO orders VALUES (?,?,?,?)",
                    (oid, rng.randint(1, 5), d.isoformat(), rng.choices(STATUSES, [70, 10, 10, 10])[0]))
        for pid in rng.sample(range(1, 41), rng.randint(1, 4)):
            con.execute("INSERT INTO order_items VALUES (?,?,?)", (oid, pid, rng.randint(1, 5)))
    con.commit()
    con.close()


def run(db: str | Path, sql: str, limit: int = 1000, timeout: float = 5.0):
    """Rows, or None on any error. mode=ro makes writes impossible; one statement only."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return con.execute(sql).fetchmany(limit)
    except sqlite3.Error:
        return None
    finally:
        con.close()


def _norm(rows):
    return sorted((tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in rows), key=repr)


def same(a, b) -> bool:
    """Execution match: same multiset of rows (order-insensitive)."""
    return a is not None and b is not None and _norm(a) == _norm(b)
