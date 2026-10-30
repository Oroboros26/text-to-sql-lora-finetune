"""Build the dataset: seeds.jsonl -> LLM expansion -> execution-validated -> split by seed group.

Usage: python -m text2sql.data
Splitting by seed group (query template) means test templates are unseen in training:
a harder, more honest test than a random row split.
"""
import json
import os
import random
from pathlib import Path

from text2sql import db
from text2sql.llm import SCHEMA, complete, extract_sql

DATA = Path("data")
N_PER_SEED = 60


def expand(seed_row: dict) -> list[dict]:
    prompt = (
        f"SQLite schema:\n{SCHEMA}\nExample:\nQ: {seed_row['question']}\nSQL: {seed_row['sql']}\n\n"
        f"Write {N_PER_SEED} new, varied question/SQL pairs of similar shape (different filters, values, "
        'phrasing). Read-only SELECT only. Reply with a JSON list of {"question": ..., "sql": ...}.'
    )
    text, _, _ = complete(os.environ["GEN_MODEL"], [{"role": "user", "content": prompt}])
    try:
        return json.loads(extract_sql(text))
    except json.JSONDecodeError:
        return []


def main() -> None:
    DATA.mkdir(exist_ok=True)
    db_path = DATA / "shop.db"
    db.build_db(db_path)
    seeds = [json.loads(line) for line in Path("data/seeds.jsonl").read_text().splitlines() if line]
    rows, seen = [], set()
    for s in seeds:
        for r in [s, *expand(s)]:
            q, sql = r.get("question", "").strip(), r.get("sql", "").strip()
            if q.lower() in seen or not db.run(db_path, sql):  # drops errors and empty results
                continue
            seen.add(q.lower())
            rows.append({"question": q, "sql": sql, "group": s["id"]})
    groups = sorted({r["group"] for r in rows})
    random.Random(42).shuffle(groups)
    n = max(1, len(groups) // 10)
    split = {"test": set(groups[:n]), "val": set(groups[n:2 * n])}
    for name in ("train", "val", "test"):
        out = [r for r in rows if (name == "train" and r["group"] not in split["test"] | split["val"])
               or (name != "train" and r["group"] in split[name])]
        (DATA / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
        print(name, len(out))


if __name__ == "__main__":
    main()
