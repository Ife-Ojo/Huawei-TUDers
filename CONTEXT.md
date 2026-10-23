COMPETITION: predict click probability on synthetic recommendation data (TechArena 2026 Phase 1). Full rules in docs/INSTRUCTIONS.md; if this file and that file disagree, that file wins.
DATA: Event = one user shown 10 candidates at one time; row = one candidate; at most one clicked=1 per event; no-click events exist. Columns: rowid, event_id, user_id, item_id, timestamp (unix s), time_bucket, day_type, history_item_ids, history_timestamps (JSON arrays: the user's last <=20 clicks before the event, oldest to newest), clicked (train only). IDs are opaque; rowid and event_id carry no information. train.csv: 686,824 events, 2025-01-20 up to before 2025-01-27, ~7.3GB. evaluation.csv: 52,939 events, one per user, that user's FIRST event at or after 2025-01-27. Some eval users are not in train; some eval histories are empty.
SUBMISSION: CSV rowid,predicted_probability, one row per eval rowid, finite values in [0,1]. 10 submissions/day/team.
METRIC: ROC AUC pooled over all rows (not per event). Only ranking matters, so use pointwise binary models, not per-event softmax or lambdarank.
VALIDATION: fit = train events 20-25 Jan UTC; val = 26 Jan UTC. Label-based stats for an event may use only labels strictly before that event's UTC day ("as-of"). Val rows use stats as of 26 Jan; eval rows use stats as of 2025-01-27. Model-training rows are 21-25 Jan (20 Jan is stats-only). Final models retrain on 21-26 Jan.
INTERFACES (everyone codes against these):
- src/validate.py: get_split() -> (fit_event_ids, val_event_ids, val_first_event_ids); report(df) with columns event_id, clicked, pred -> dict {auc_first, auc_all}.
- src/features.py: build_features(event_ids, asof_date) -> Polars DataFrame with rowid, event_id, clicked, then feature columns; FEATURE_COLS list.
- src/embeddings.py: embedding_features(rowids_or_events, asof_date, mode) -> Polars DataFrame keyed by rowid.
- Model outputs: outputs/val_pred_<model>.parquet and outputs/eval_pred_<model>.parquet, both with columns rowid, pred. <model> is lgbm or nn.
- Data in data/processed/: events.parquet, cands.parquet and the ID maps (built by src/prep.py).

AGENT RULES:
1. Read CONTEXT.md before every task.
2. Never load train.csv fully into pandas. Use Polars lazy/streaming or DuckDB.
3. Never build a feature from labels of the row being predicted or from later time.
4. Do not edit EXPERIMENTS.md. Append your result rows to docs/log/<your-name>.md. Only P1 merges them.
5. Work on a branch named after the task. When done: run it, show me the numbers, commit as "type: what (val AUC x)", push the branch, do NOT merge. Before committing, make sure no file over 50MB is staged. Touch only files you own.
6. Keep code simple. No frameworks or abstractions I did not ask for.
