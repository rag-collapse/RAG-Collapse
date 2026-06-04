# all_experiments — consolidated RAG-collapse data

Single copy of the experiment JSONs, gathered from the per-user output trees
under `/work/pi_dagarwal_umass_edu/project_4/file_storage/`, plus **reranker-only
gap-fill from scratch** (`/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/
{oyilmazel,ffatima}_umass_edu`). Originals are untouched — these are **copies**.

## Scope (filtered)

- **Models (4 only):** `Qwen2.5-14B-Instruct`, `DeepSeek-R1-Distill-Qwen-7B`,
  `Llama-3.1-8B-Instruct`, `Mistral-7B-Instruct-v0.3` — the gepa_pipeline set.
  Qwen2.5-7B/1.5B, Qwen3-4B, Qwen3.5-9B and gpt-oss-20b are excluded.
- **rsenapati's runs:** only `agentic_rag` is kept (his focus area); his
  baseline/search duplicates of other people's runs are dropped.
- **Canonical baselines:** ffatima for Qwen-14B & Mistral; ratirastogi for
  Llama & DeepSeek. oyilmazel's graphite baselines were stale and are dropped
  (his HotpotQA data is the sole copy and is kept).
- **Reranker:** the off-the-shelf λ-sweep lives in `/work` (Qwen-14B); the
  oracle / desklib / finetuned ablations and the non-Qwen reranker eval/entity
  live only in scratch and are pulled in as **reranker-only gap-fill** (a scratch
  file is added only if its dataset+model+tree+filename isn't already in `/work`).
  Coverage is intentionally partial — not every model has every eval/entity.
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
- `UNREADABLE.txt` — only written when some source files can't be read
  (permissions). Currently absent → every in-scope file was copied successfully.

## Regenerating

```bash
cd ~/consolidate          # holds consolidate_experiments.{py,sh}
sbatch consolidate_experiments.sh
```

The script (repo: `scripts/consolidate_experiments.py`) is idempotent — it
skips any dest file that already exists with the same size. If you change the
layout scheme, clear `graphite/` and `hotpotqa/` first, then re-run.
