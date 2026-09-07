# Handoff: Option B citation program — DeepSeek agentic is the only remainder

The Option B rebuttal work is complete for three of the four models. Everything is committed and pushed
to `main` (latest `474d0c9`). One cell of the cross-model table is still empty: DeepSeek agentic. This
file says what is done, the single task that remains, the exact recipe, and the caveats. The deliverable
is `docs/citation_attribution_spec.md`. It holds every result.

Commit trailer for this repo:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

## What is done and committed

The over-citation program regenerates real leave-one-out (LOO) citations for every model and variant
that never logged them, then runs the over-citation ratio and R3 (query-alignment control) on them.
Plain-RAG generator is `~/loo_attribution.py`. Agentic generator is `~/agentic_loo.py` (faithful
tool-calling, Option A over the recorded store). Ratio folder is
`scripts/camera_ready/overcite_ratio.py`. R3 is `~/r3_from_sidecar.py`. HotpotQA is excluded on
purpose. Its 1-to-3-token answers make the lexical change score degenerate.

Cross-model coverage of the Option B table in `docs/citation_attribution_spec.md` (ratio and R3 both):

| regime | Qwen2.5-14B | Llama-3.1-8B | Mistral-7B | DeepSeek-R1-7B |
|---|---|---|---|---|
| baseline (RA/RO/Search) | done | done | done | done |
| paraphrase (RA/RO/Search) | done | done | done | done |
| rerank (λ=0.7 oracle/desklib) | done | done | done | done |
| agentic RAG | done | done | done | **remaining** |

This session's four commits: `d09ccf7` (para+rerank ratios), `e25c756` (para+rerank R3), `b933441`
(agentic ratios), `474d0c9` (agentic R3). Sidecars live on Unity in `~/loo_cite_*.json` and are archived
in `~/rag_rebuttal_scripts/`. Derived artifacts are committed under `camera_ready_outputs/loo_citations/`
(the two `overcite_ratio_*.tsv` and the two `r3_*_results.txt`).

## The one remaining task: DeepSeek agentic

DeepSeek's original agentic run was degenerate (84% empty answers) because it was served with
`max_tokens=512`, which truncated DeepSeek-R1's long reasoning before the final answer. So a full agentic
re-run comes first, then the agentic LOO pass over that re-run.

**The re-run is sharded across 8 parallel jobs.** A single 400-question agentic_rag run needs ~80 to 150h
for 30 rounds and `pipeline.py` writes its output only once at the end (no checkpointing), so a single job
cannot fit the 48h gpu wall. The first attempt (job 64025428) was killed at iter 17/30 with no output for
exactly this reason. Because each question owns an independent retrieval store ([pipeline.py:686](../pipeline.py)),
the run is split into 8 shards of 50 questions and submitted in parallel. Each shard runs the full 30
rounds in ~10 to 19h, well inside 48h. Dataset shards are `~/ds_agentic_shard_00..07.jsonl`, the job is
`~/run_ds_agentic_shard.sh` (submitted as `sbatch --export=ALL,SHARD=NN`), jobs **64057656 to 64057663**.
Each shard uses the validated 2-GPU setup (DeepSeek answer server + Qwen2.5-7B doc server) with the
`max_tokens 4096` fix already in the pipeline call, unique ports derived from the job id, and a
served-name readiness check on both servers.

**Dependency — Phase R (run).** Wait until all 8 shards print `### shard NN done ###` and write
`~/deepseek_agentic_rerun_shard{00..07}.json`. Verify each is clean (out file prints
`server <port> up serving deepseek-r1-distill-qwen-7b`, `does not exist` 404 count 0). Then merge into the
single 400-question run:
```
python3 ~/merge_shards.py ~/deepseek_agentic_rerun.json ~/deepseek_agentic_rerun_shard{00,01,02,03,04,05,06,07}.json
```
(`~/merge_shards.py` concatenates the disjoint `questions` arrays and keeps one metadata block; confirm it
reports 400 questions.) Only then run the agentic LOO pass below.

### Step 1 — patch the max_tokens truncation (critical, do not skip)

`~/agentic_loo.py` line 25 hardcodes `max_tokens=512`:
```
llm, _ = build_llm(model_mode="server", model_name=MODEL, temperature=0.7, max_tokens=512, ...)
```
For DeepSeek this is the exact bug that made the original run degenerate. It truncates the reasoning
trace before the answer. Bump it to `max_tokens=4096` before the DeepSeek pass (smoke confirmed the fix
takes the empty rate from 92% to 17%). The other three models already ran at 512 without harm, so change
it only for this DeepSeek run (edit, run, and note it, or add a small env override).

### Step 2 — create and submit the DeepSeek agentic job

There is no `run_agloo_deepseek.sh` yet. Copy `~/run_agloo_qwen.sh` (it carries the collision fixes from
this session, see below) and change:
- serve line: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, `--served-model-name deepseek-r1-distill-qwen-7b`,
  and the DeepSeek flags `--reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser hermes`.
- `SERVED=deepseek-r1-distill-qwen-7b` (so the readiness check greps the right name).
- run JSON = `$HOME/deepseek_agentic_rerun.json` (the merged 400q run from the shard merge above, NOT the
  old all_experiments file).
- out = `$HOME/loo_cite_deepseek_agentic.json`.
- keep `--max-model-len 16384`, the unique `PORT=$((20000 + SLURM_JOB_ID % 10000))`, and the `/v1/models`
  readiness check unchanged. Keep `-t 48:00:00`.

Submit, then verify the run is clean the same way this session did: the out file prints
`server up on <port> serving deepseek-r1-distill-qwen-7b`, the `does not exist` 404 count stays 0, and
the final sidecar empty rate is in the single-to-low-double digits (not 60%+). The 7B is ~4h like the
others.

### Step 3 — fold the two DeepSeek agentic cells

Both empty cells are marked `(re-running)` in `docs/citation_attribution_spec.md` under
"Option B Phase 2b".

1. **Over-citation ratio.** Round-1 and pooled:
   ```
   AG=/work/.../all_experiments/graphite/agentic_rag/experiment_outputs   # not needed; use the rerun JSON
   python3 ~/overcite_ratio.py "$HOME/deepseek_agentic_rerun.json|$HOME/loo_cite_deepseek_agentic.json|deepseek_agentic"
   ```
   Write the round-1 ratio into the DeepSeek column of the "Agentic RAG (round-1)" table. Update
   `camera_ready_outputs/loo_citations/overcite_ratio_agentic.tsv` with the row.
2. **R3.** Add the DeepSeek agentic pair to a copy of `~/run_r3_agentic.sh` (lean cpu-preempt job, ~6 min)
   and submit. Write the self_gen A→B result into the DeepSeek column of the agentic R3 table. Append the
   output to `camera_ready_outputs/loo_citations/r3_agentic_results.txt`.
3. Commit and push both. The agent cannot push from Unity, so commit and push from the laptop.

## Lessons from this session (reuse them, do not repeat the bug)

- **The agentic vLLM jobs must use a unique port.** The first agentic batch was corrupted (empty rates
  64/73/98%) because all three jobs were co-scheduled on one node and served vLLM on the same
  `--port 8000`. `ServerLLM` auto-discovers the served model from its endpoint
  (`llm_service/server_llm.py:113`), so every client latched onto one server and 404-stormed. The fix,
  already in `~/run_agloo_qwen.sh`: `PORT=$((20000 + SLURM_JOB_ID % 10000))` used in both `vllm serve
  --port` and the client api_base, plus a `/v1/models` readiness check that greps the expected served
  name. See memory `agentic-loo-vllm-job-pitfalls`.
- **Cap `--max-model-len 16384`.** The 14B failed vLLM engine init on a small vram40 card because the
  default 32k context left too little KV cache. The cap fixes it with room to spare and the agentic
  context is tiny.
- **CPU-queue congestion.** The first R3 job sat 6h on `(Priority)` in partition `cpu`. Submitting lean
  (`-p cpu-preempt -c 8 --mem 24g -t 02:00:00`) scheduled in minutes. R3 is a 5-to-40 min job.

## Caveats to carry into the write-up

- **Agentic LOO is Option A**, the faithful tool-calling flow over the recorded store. The store at round
  r is reconstructed from the documents the run recorded through round r, not the full growing store
  (that is not saved). Say this plainly. It is the only tractable faithful attribution of the existing run.
- **The over-citation ratio is partly a prefilter artifact.** The top-2 overlap gate over-selects self-gen
  by ancestry (2 to 2.9x). Lead the §6 rewrite with the counterfactual and R7, not the raw share.
- **Rerank neutralizes the per-doc effect; paraphrase does not.** Rerank pooled ratio ≈1.0 and null R3
  self_gen; paraphrase over-citation survives and grows under controls on both Replace-One and Search.
  This is the mitigation contrast to lead with.

## Last step for the whole markdown pass

`docs/citation_attribution_spec.md` is the one markdown file not yet deslopped. Once the DeepSeek agentic
cells are folded in and it is stable, deslop it per the poteto writing standard (remove long dashes and
mid-sentence colon connectors, split run-ons), preserving every number, path, and table. That closes the
"deslop every markdown file" task. The Option B sections added this session are already written clean, so
the deslop is mostly the older top half of the file.

## Environment notes

- `ssh unity` works. Activate the env with
  `source /modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1/etc/profile.d/conda.sh; conda activate ragenv`.
- Scripts that import the repo (`agentic_loo.py`) run from `~/RAG-Collapsement-on-Self-Refined-Generation`
  with `export PYTHONPATH=$HOME/RAG-Collapsement-on-Self-Refined-Generation`.
- HF cache: `export HF_HOME=/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hf_cache`
  with `HF_HUB_OFFLINE=1`.
- The agent cannot push from Unity. Commit and push from the laptop. Never commit API keys.
- Job scripts and outputs are archived in `~/rag_rebuttal_scripts/` on Unity.
