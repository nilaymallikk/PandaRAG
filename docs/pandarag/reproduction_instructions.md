# Reproduction instructions

```bash
# 0. environment (Python 3.14 is the supported interpreter for this venv)
ln -sf /usr/bin/python3.14 .venv/bin/python
python -m pytest tests/ -q                     # expect 298 passing

# 1. corpora + indexes (local, no LLM calls; ~15 min for the embeddings)
python -m src.corpus_retrieval.corpus_cli
python -m src.corpus_retrieval.index_cli

# 2. corpus retrieval sanity metrics
python -m src.corpus_retrieval.benchmark --dataset hotpotqa
python -m src.corpus_retrieval.benchmark --dataset 2wiki
python -m src.corpus_retrieval.benchmark --dataset musique

# 3. full experiment matrix (resumable; safe to re-run after interruptions)
python -m src.experiments.run_matrix --concurrency 3

# 4. ablations on the frozen stratified subset
python -m src.experiments.run_ablations --sample-size 100

# 5. faithfulness judging (frozen judge protocol, stratified sample)
python -m src.analysis.faithfulness_scope_b --sample 100

# 6. analysis + documents
python -m src.analysis.figures
python -m src.analysis.report
```

Notes:

- Every run resumes: re-running a command fills only missing/failed questions. Failures
  stay in the JSONL as `success: false` rows and are never silently dropped.
- The LLM disk cache means a second run of the same configuration is replayed at zero
  API cost; those records carry `from_cache: true` and are excluded from `live` spend.
- Corpora and indexes live under `cache/` (gitignored); their fingerprints are recorded
  in `docs/pandarag/phase1_artifacts.json`.
