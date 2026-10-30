"""Score one system on data/test.jsonl and write eval/results/<name>-<timestamp>.json.

python eval/run.py --name qwen-base-0shot --model Qwen/Qwen2.5-Coder-1.5B-Instruct --shots 0
Env: LLM_BASE_URL, LLM_API_KEY, PRICE_IN / PRICE_OUT (USD per 1M tokens; for self-hosted
models derive from your GPU $/hour and measured throughput, and say so in the README).
"""
import argparse
import json
import os
import re
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from text2sql import db
from text2sql.llm import complete, extract_sql, messages

DB = Path("data/shop.db")
READ_ONLY = re.compile(r"\s*(select|with)\b", re.IGNORECASE)


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def pick_shots(train: list[dict], k: int) -> tuple[tuple[str, str], ...]:
    seen, shots = set(), []
    for r in train:  # first row of each of the first k groups: deterministic
        if r["group"] not in seen and len(shots) < k:
            seen.add(r["group"])
            shots.append((r["question"], r["sql"]))
    return tuple(shots)


def norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def one(args, shots, row):
    t = time.perf_counter()
    text, tin, tout = complete(args.model, messages(row["question"], shots))
    lat = time.perf_counter() - t
    sql = extract_sql(text)
    gold, pred = db.run(DB, row["sql"]), db.run(DB, sql)
    return {"lat": lat, "tin": tin, "tout": tout, "exec": db.same(gold, pred), "valid": pred is not None,
                "exact": norm(sql) == norm(row["sql"]), "ro_violation": not READ_ONLY.match(sql)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    test, shots = load("data/test.jsonl"), pick_shots(load("data/train.jsonl"), args.shots)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda r: one(args, shots, r), test))
    wall = time.perf_counter() - t0
    n, lats = len(res), sorted(r["lat"] for r in res)
    cost = sum(r["tin"] * float(os.getenv("PRICE_IN", "0")) + r["tout"] * float(os.getenv("PRICE_OUT", "0"))
               for r in res) / 1e6
    out = {"name": args.name, "model": args.model, "shots": args.shots, "n": n,
               "exec_acc": sum(r["exec"] for r in res) / n, "exact_match": sum(r["exact"] for r in res) / n,
               "valid_sql": sum(r["valid"] for r in res) / n, "ro_violations": sum(r["ro_violation"] for r in res),
               "p50_s": st.median(lats), "p95_s": lats[int(0.95 * (n - 1))], "throughput_rps": n / wall,
               "cost_per_1k_usd": cost / n * 1000, "workers": args.workers}
    path = Path("eval/results") / f"{args.name}-{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
