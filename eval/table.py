"""Print the latest result per system as a markdown table for the README."""
import json
from pathlib import Path

latest = {}
for f in sorted(Path("eval/results").glob("*.json")):  # timestamped names sort chronologically
    r = json.loads(f.read_text())
    latest[r["name"]] = r
cols = ["exec_acc", "exact_match", "valid_sql", "ro_violations", "p50_s", "p95_s", "throughput_rps", "cost_per_1k_usd"]
print("| system | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
for name, r in latest.items():
    print(f"| {name}{' (SYNTHETIC)' if r.get('synthetic') else ''} | " + " | ".join(f"{r[c]:.3g}" for c in cols) + " |")
