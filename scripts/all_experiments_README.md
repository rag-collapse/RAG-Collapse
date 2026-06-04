# all_experiments — consolidated RAG-collapse data

Single copy of the experiment JSONs, gathered from the per-user output trees
under `/work/pi_dagarwal_umass_edu/project_4/file_storage/`. Originals are
untouched — these are **copies**.

## Scope (filtered)

- **Models (4 only):** `Qwen2.5-14B-Instruct`, `DeepSeek-R1-Distill-Qwen-7B`,
  `Llama-3.1-8B-Instruct`, `Mistral-7B-Instruct-v0.3` — the gepa_pipeline set.
  Qwen2.5-7B/1.5B, Qwen3-4B, Qwen3.5-9B and gpt-oss-20b are excluded.
- **rsenapati's runs:** only `agentic_rag` is kept (his focus area); his
  baseline/search duplicates of other people's runs are dropped.
- **Graphite = full 400-question benchmark only.** Files are filtered by actual
  question count, so the 50-question runs and stray smoke/debug runs are excluded.
  (HotpotQA keeps its own dataset size.)

## Layout

```
all_experiments/
  <dataset>/<method>[/<config>]/<output_tree>[/<owner>]/<org>/<model>/.../<file>
```

- **dataset**: `graphite` | `hotpotqa`
- **method[/config]**:
  - `baseline/{replace_all, replace_one, search}` — the core collapse simulations
  - `paraphrase/{search, replace_one, hybrid}` — paraphrase mitigation
  - `agentic_rag` — agentic RAG
  - `rerank` — reranker (λ sweep + ablations, distinguished by filename)
  - `misc` — anything else (e.g. `hotpot_train.json`)
- **output_tree**: `entity_extraction_output` (collapse), `evaluation_outputs`
  (ROUGE/TES/etc.), `experiment_outputs` (raw per-round generations)
- **owner**: `ffatima` | `ratirastogi` | `rsenapati` | `oyilmazel` | `reranker`
  — **present ONLY when more than one person ran the exact same file.** Unique
  runs are stored flat (no owner level). So for most methods (agentic_rag, etc.)
  there is no owner folder; for the triplicated Qwen baselines you'll see
  `.../ffatima/...`, `.../rsenapati/...`, `.../oyilmazel/...` side by side.

For the canonical (reported) baseline, **ffatima** is the reference run for the
Qwen/Mistral models; **ratirastogi** for DeepSeek/Llama. See
`docs/data_locations.md` in the repo for the exact per-model/per-variant choices.

## Files

- `MANIFEST.csv` — provenance: `src,dest,dataset,method,output_tree,owner,kind`
  for every copied file (`kind` = `dup` or `unique`).
- `UNREADABLE.txt` — 24 source files that could **not** be copied because the
  owner's umask left them non-group-readable. All are `ffatima` raw
  `experiment_outputs/` + `.checkpoint.json` files for Qwen-14B and Mistral.
  **No eval or entity files are affected**, so the full collapse-analysis set is
  present. To recover them, ask Fabeha to run:
  ```bash
  chmod -R g+rX /work/pi_dagarwal_umass_edu/project_4/file_storage/ffatima_umass_edu/experiment_outputs
  ```
  then re-run the consolidation (idempotent — only the missing files are fetched).

## Regenerating

```bash
cd ~/consolidate          # holds consolidate_experiments.{py,sh}
sbatch consolidate_experiments.sh
```

The script (repo: `scripts/consolidate_experiments.py`) is idempotent — it
skips any dest file that already exists with the same size. If you change the
layout scheme, clear `graphite/` and `hotpotqa/` first, then re-run.
