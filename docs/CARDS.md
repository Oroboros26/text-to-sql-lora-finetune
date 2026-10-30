# Model card (publish as README.md of the HF model repo)

- **Base model:** see `train/config.yaml`. **Method:** QLoRA (r/alpha/dropout from config), seed 42.
- **Task:** question -> one read-only SQLite query for the schema in `src/text2sql/schema.sql`.
- **Results:** copy the table from the README; include the baselines.
- **Limits:** one fixed schema, synthetic data, small test set; not for other schemas or untrusted production SQL.
- **Training compute:** GPU, hours (from `train_stats.json`).

# Dataset card (publish as README.md of the HF dataset repo)

- **Creation:** 30 hand-written seed pairs (`data/seeds.jsonl`), expanded by `GEN_MODEL` (name it here), kept only if gold SQL executes with non-empty results on the synthetic DB, deduped by question.
- **Splits:** train/val/test by seed group; test frozen before any training. Sizes: fill in from `make data` output.
- **Hand check:** record the sample size and error rate of the manual review (target 10%).
- **PII:** none; the DB is randomly generated.
