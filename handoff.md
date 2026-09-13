# Handoff: RAG-collapse / HotpotQA work (fresh-start context)

Everything below is built, reviewed, and smoke-validated. The only thing not yet done is the
full-scale HotpotQA runs. This doc covers three things to know, in order. First the `all_experiments`
data folder. Second the HotpotQA experiments. Third working with Unity. The repo and code map and the
next steps follow.

- Repo: `RAG-Collapse`. Research on inference-time RAG collapse. A model
  answers from retrieved docs, its answers become "documents" fed back next round, and over rounds the
  answers degenerate and drift. Read `CLAUDE.md` for the repo map, `docs/misinfo_error_compounding.md`
  for the experiment, and `docs/hotpotqa_smoke_results.html` for the beginner primer and all smoke results.
- Branch: `misinfo-error-compounding` (draft PR #32 into `main`). Latest `origin` is about `b76fced`.
- Beginner orientation: open `docs/hotpotqa_smoke_results.html`. It has a "START HERE" primer and a glossary.

---

## 1. The `all_experiments` folder (consolidated data on Unity)

Path: `/work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments/`

Layout: `<dataset>/<method>[/<config>]/<output_tree>[/<owner>]/<org>/<model>/<file>`, where
`<output_tree>` is one of `experiment_outputs` (raw run JSON), `evaluation_outputs` (text metrics), or
`entity_extraction_output` (entity metrics). Built by `scripts/consolidate_experiments.py`. See
`docs/data_locations.md` and `scripts/all_experiments_README.md`.

- Datasets: `graphite` (the umass-entity set) and `hotpotqa`. Methods: `baseline`, `agentic_rag`,
  reranker ablations, and others.
- Canonical baseline owners, meaning which owner's run to trust per model: ffatima owns Qwen-14B and
  Mistral, Rati owns Llama and DeepSeek, rsenapati owns Agentic RAG.
- This session regenerated the missing DeepSeek graphite baseline raw outputs. They were never saved to
  shared storage and the original raw was lost. They are now present and verified clean (400 q, 240,000
  answers, 0 byte-artifacts):
  `all_experiments/graphite/baseline/{replace_all,replace_one,search}/experiment_outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/local_<variant>.json`
  - Caution. The `evaluation_outputs/` and `entity_extraction_output/` for the DeepSeek baseline there
    are STALE (dated Apr, computed from the lost original raw). They do not match the new raw (Jun).
    Re-run `evaluation.py` and `entity_extraction.py` on the new raw to make the derived metrics consistent.
  - A laptop copy of the three raw files lives in `baseline-DeepSeek-R1-Distill-Qwen-7B/` (gitignored).
- HotpotQA experiment outputs land per-owner, not in all_experiments by default:
  `…/file_storage/rsenapati_umass_edu/hotpotqa_distractor_experiment/<served>/` (misinfo and synthetic),
  and `…/rsenapati_umass_edu/base_hotpotqa_distractors/native_<model>/` (native). Consolidate later if wanted.

---

## 2. The HotpotQA experiments (three of them, all smoke-validated, full runs pending)

All three use the recursive loop in `hotpot_pipeline.py` (variants `search` 30 rounds, `replace_one`
20, `hybrid` 10) with FAISS/E5 Wikipedia retrieval and a Qwen2.5-7B doc/judge model. Diverse-aware and
injection metrics live in `hotpot_evaluation.py`. Launchers and the RUNBOOK are in
`scripts/hotpot-misinfo/` and `base_hotpotqa_distractors/`.

### 2a. Misinfo error-compounding (`scripts/hotpot-misinfo/`)
The doc synthesizer injects a false claim into the generated doc each round, and tracks whether it
propagates. Flags: `--doc-synthesis-mode {faithful,counterfactual,freeform}`, `--target-mode
{final_answer,intermediate_hop,untargeted}`, `--inject-round`. Produces a per-question `injection`
record and ASR metrics. Smoke passed (parity, counterfactual, freeform discovery, non-degenerate ASR).
Full run not done. Per-model launchers are `launch_{qwen14b,mistral7b,llama8b,deepseek7b}.sh`.

### 2b. Synthetic round-0 distractors, "Option B" (`base_hotpotqa_distractors/`, `launch.sh`)
Corrupt a fraction of round-0 retrieved docs into DIVERSE wrong-answer distractors, each a distinct
invented falsehood. Flags: `--distractor-fraction`, `--distractor-mode {rewrite,substitution,native_noise}`.
The Q1 fix is done (gold-leak exclusion plus substitution fallback in
`pipeline/misinfo.py::_apply_rewrite`). Re-smoke confirmed 0 gold-leak, 2.0 distinct entities/q, 15/15
effective. A qwen14b sweep was started then stopped when the user clarified the real goal is Option A
(below). Code and partial outputs remain.

### 2c. Native HotpotQA distractor setting, "Option A", THE CURRENT FOCUS (`base_hotpotqa_distractors/`, `launch_native.sh`)
Replicates the original HotpotQA paper (Yang et al., EMNLP 2018) distractor setting. Round-0 context is
each question's native 2 gold plus 8 TF-IDF distractor paragraphs (answer-absent hard negatives, not
wrong-answer assertions), run through the recursive loop. Flags: `--initial-docs native_distractor`
(default `faiss`, byte-identical) and `--distractor-gold-only` (the control). Eval is F1/EM/`gold_match`
over rounds. The headline is distractor-setting vs gold-only (the paper's ablation).
- Smoke-validated across all 4 models (qwen14b, mistral-7b, llama-3.1-8b, deepseek-r1-7b). Faithful
  round-0 is 10 docs (2 gold plus 8 distractor), gold-only is 2 gold, and answers are clean (DeepSeek included).
- Use `replace_one` (default) or `hybrid`. `search` makes the universe just the 10 native paragraphs
  (documented caveat). `replace_one` keeps all 10, `hybrid` truncates to `num_db_docs`.
- Caution. DeepSeek scored about 0 gold_match in the smoke. Answers were clean but diverse and did not
  match the short gold spans, a reasoning-model answer-format issue. Investigate before trusting DeepSeek
  native numbers.

Run the full native experiment (the next step). On Unity, after `git pull`:
```bash
bash base_hotpotqa_distractors/smoke_native.sh                  # fast re-check (downloads distractor file if missing)
bash base_hotpotqa_distractors/launch_native.sh                 # qwen14b: distractor-setting + gold-only, replace_one, 400q, num_runs 10
# other models: distinct ports + model flags (see launch_native.sh header / §3 example):
ANSWER_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct ANSWER_SERVED=llama3.1-8b ANSWER_MAX_NUM_SEQS=128 \
  ANSWER_PORT=5190 DOCGEN_PORT=5191 OUTDIR=$HOME/native_llama bash base_hotpotqa_distractors/launch_native.sh
```

---

## 3. Working with Unity (hard-won lessons, read before touching the cluster)

- GitHub auth: the agent CANNOT `git pull` or `push` on Unity (HTTPS remote, no stored credential). The
  workflow all session was to commit and push from the laptop clone, then the user pulls on Unity with a
  PAT. Do not try to push from Unity.
- Conda/Python: the Unity env is `ragenv` (vllm/faiss/torch). `module load conda; conda activate ragenv`
  works only in a login shell (`ssh unity 'bash -lc "…"'` or `bash -ls`). In a non-login `bash -s`
  heredoc, `module` and `conda` are undefined, so call the env python directly:
  `$HOME/.conda/envs/ragenv/bin/python`. On the laptop the env is `rag-collapse`, python
  `/c/Users/riddh/anaconda3/envs/rag-collapse/python.exe` (no GPU, so `py_compile` / pytest / wiring only).
- Launchers run detached: `setsid env … bash <launcher>.sh > log 2>&1 </dev/null &` from a login shell.
  The launcher submits the 2 servers, polls until they serve (`curl /models`), submits the client plus a
  reaper (`--dependency=afterany` to `scancel` the servers when the client finishes), then exits. It
  persists after SSH closes. Monitor by reading the log and `squeue --me`.
- SLURM 2-day cap: servers request 48h but the reaper kills them early, and clients request 47h (servers
  start about 10 min earlier, so they outlive clients). For smokes, pass short `SERVER_TIME=01:30:00
  CLIENT_TIME=01:00:00`.
- Ports must be distinct per concurrent run (the launchers expose `ANSWER_PORT`/`DOCGEN_PORT` and now
  forward them to the servers). In use: graphite baseline `5154/5153`, misinfo per-model `5164–5170`,
  synthetic distractor `5180/5181`, native per-model `5186/5187` (qwen) and `5188–5193` (mistral/llama/deepseek).
- GPU queue is volatile. It reached 1500+ pending at one point and 52 idle at another. Short server-time
  jobs backfill fast, and long 48h jobs are sometimes still scheduled instantly when GPUs free up.
- vLLM 0.20.0 DeepSeek detok bug: DeepSeek-R1 `message.content` comes back as byte-level BPE (`Ġ` is a
  space, `Ċ` a newline), and `--reasoning-parser deepseek_r1` does not fix it (it is a detok regression;
  CMU/HF issues #12954/#19222/#24459 do not match it). Fixed client-side in
  `llm_service/server_llm.py::_clean_response` (byte-decode when markers present, plus strip `<think>`).
  Serve DeepSeek WITHOUT `--reasoning-parser`. Mistral needs `--tokenizer-mode mistral`. Non-agentic
  variants need no tool-call flags.
- HotpotQA data host `curtis.ml.cmu.edu` is DEAD. `base_hotpotqa_distractors/fetch_distractor_file.py`
  reconstructs `hotpot_dev_distractor_v1.json` from HuggingFace (`hotpot_qa`, distractor/validation,
  7,405 q). It is already generated at
  `/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_distractor_v1.json`.
- Scripting-over-SSH gotchas (these wasted time). (a) Parens in `echo` break `bash -c "…"`, so avoid
  `(…)` in echoed strings. (b) `pkill -f <pattern>` can match its own SSH shell's command line and
  self-kill (exit 255), so use read-only `pgrep`/`ps` or PID-specific kills. (c) Long-running SSH polls
  drop (exit 255), so run monitors via the Bash tool's `run_in_background` (or detached) and verify state
  directly afterward. (d) For nested-heredoc and quoting, prefer `ssh unity 'bash -ls' <<'EOF'` (login
  shell, literal heredoc) when you need `sbatch`/`module` plus env vars with spaces.
- Data artifacts (Unity scratch `…/oyilmazel_umass_edu-rag_collapse/`): `hotpotqa_index/{ivf.index,
  docid_map.json}`, `hf_cache/`, `hotpot_dev_fullwiki_v1.json` (fullwiki, gold often absent), and
  `hotpot_dev_distractor_v1.json` (the paper's distractor setting, generated this session). `CACHE_DIR`
  in SLURM scripts stays hardcoded to this scratch path, not `$USER`. Never commit API keys.

---

## 4. Code map (as-built, reference symbols not line numbers)

- `hotpot_pipeline.py`: round loop, flags and guards (`distractor_enabled`, `native_seed`, both
  default-off and byte-identical), `--initial-docs {faiss,native_distractor}`, `--distractor-fraction`,
  `--distractor-mode`, `--distractor-gold-only`, and `DistractorController.prepare_and_apply` at round 0.
- `pipeline/misinfo.py`: `MisinfoController` (injection), `DistractorController` plus `_apply_rewrite`
  (with the gold-leak/fallback fix) plus `native_context_docs` (native gold/distractor split), and
  helpers (`substitute_in_text`, `choose_distinct_substitutes`, `validate_substitute`, `normalize`, `load_*`).
- `hotpot_evaluation.py`: F1/EM plus `injection_aggregates` plus `distractor_aggregates` (all no-op
  without records).
- `formatters.py`: synthesis/freeform/rewrite-distractor prompts and judges.
- `llm_service/server_llm.py`: `_clean_response` (byte-decode plus `<think>` strip) on every answer path.
- Launchers: `scripts/hotpot-misinfo/{launch_<model>.sh, launch_all_variants.sh, run_variant_client.sh,
  server_answer.sh, server_docgen.sh, smoke_*.sh, RUNBOOK.md}`, `scripts/deepseek_baseline/`, and
  `base_hotpotqa_distractors/{launch.sh, run_sweep.sh, launch_native.sh, run_native.sh, smoke*.sh,
  compare_*.py, fetch_distractor_file.py, README.md}`.
- Tests: `tests/test_misinfo.py` plus `base_hotpotqa_distractors/test_{base_distractors,native_seed}.py`
  (32 pass in `rag-collapse`, pure-Python, no GPU).
- Docs/HTML: `docs/hotpotqa_smoke_results.html` (results plus beginner primer plus glossary),
  `docs/hotpotqa_experiments.html`, `docs/original_vs_distractor.html`, `docs/distractor_experiment.html`.

---

## 5. Open next steps

1. Full native runs (the user's focus): `replace_one`, 400 q, num_runs 10, distractor-setting plus
   gold-only, per model (qwen14b validated, mistral/llama/deepseek smoke-clean). Distinct ports and model
   flags per §2c. GPUs permitting, run models in parallel.
2. Investigate DeepSeek's ~0 gold_match under native seeding (answer-format vs short gold spans) before
   trusting its numbers. It is possibly an answer-extraction/normalization issue for reasoning output.
3. Refresh the stale DeepSeek baseline eval/entity in `all_experiments` (re-run `evaluation.py` and
   `entity_extraction.py` on the regenerated raw).
4. Optional: the synthetic distractor full sweep (Option B), and the "introduce distractor, remove it,
   does the error persist?" compounding experiment.
5. Everything commits from the laptop, and the user pulls on Unity. Keep default-off flags byte-identical.
