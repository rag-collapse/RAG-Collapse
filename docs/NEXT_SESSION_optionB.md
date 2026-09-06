# Handoff: Option B citation program (fresh-start context)

Continue the camera-ready rebuttal work. Everything through Phase 2a is committed to `main`. Five
Unity jobs are still running. This file says what is done, what is running, how to collect each
result, and what to write into the spec doc. Read `docs/citation_attribution_spec.md` first. It is
the deliverable and holds every result.

Commit trailer for this repo:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

## What the program is

Reviewers questioned the §6 citation attribution. The reported method is leave-one-out (LOO)
counterfactual, not pure overlap. "Option B" generates real LOO citations for every model and
variant that never logged them, then runs the over-citation ratio and R3 (query-alignment control)
on all of them. The generator is `~/loo_attribution.py` on Unity (plain-RAG variants) and
`~/agentic_loo.py` (agentic, faithful tool-calling). Both validated. HotpotQA is excluded on
purpose. Its 1-to-3-token answers make the lexical change score degenerate.

## Done and committed to `main`

C1, C2, C3. R1 (placebo), R2 (threshold sweep), R3 (query-alignment, all four models via
`r3_from_sidecar`), R3b (HotpotQA retrieval alignment, all three variants), R4 (collapse vs
contamination), R5 (per-model dF1 with CIs), T1 (direct-vs-LOO on same items, round-1 ratio 2.57 vs
2.49), T2 (extractor-agreement kit), F1 (number sweep), F2 (hAN7 conditions), R7 (novelty
restriction, token and entity, both near-null), the prefilter-bias finding (top-2 gate over-selects
self-gen 2-2.9x across all models), the overlap-family cross-model sweep (`allruns_overlap.py`), the
Phase 2a real-LOO baseline citations for all four models, and cross-model R3. All of `scripts/camera_ready/`
and the sidecars in `camera_ready_outputs/loo_citations/` are committed. Every markdown file except
`docs/citation_attribution_spec.md` was deslopped (see the deslop commits).

## Running on Unity now (check with `squeue -u $USER`)

| Job | Name | What it makes | Walltime left |
|---|---|---|---|
| 64015742 | loobc-deepseek | DeepSeek paraphrase + rerank LOO citations | ~12h, nearly done |
| 64025428 | ds-agentic-rerun | Full DeepSeek agentic experiment RE-RUN (fixed config) | ~42h of 48h |
| 64025426 | agloo-qwen | Qwen agentic LOO (Option A) | ~6h, will NOT finish 400q |
| 64025427 | agloo-llama | Llama agentic LOO | ~4h, will NOT finish 400q |
| 64025431 | agloo-mistral | Mistral agentic LOO | ~4h, will NOT finish 400q |

Sidecars land as `~/loo_cite_<model>_<variant>.json` on Unity. Paraphrase and rerank are done for
Llama, Mistral, Qwen. DeepSeek's last file `loo_cite_deepseek_rerank_0.7_oracle.json` is the only
paraphrase/rerank piece still pending.

## The agentic LOO jobs will time out. Resume them.

Agentic LOO is ~5 min per question for 400 questions, roughly 33h per model. The current jobs have
10-to-12h walltimes, so each will checkpoint out partway. `~/agentic_loo.py` checkpoints per question
and resumes (it skips questions already in the output). When a job ends before printing
`### agentic loo done ###`, resubmit its script to resume. Bump the walltime first so it finishes in
one more pass:
```bash
sed -i 's/#SBATCH -t .*/#SBATCH -t 48:00:00/' ~/run_agloo_qwen.sh    # and _llama, _mistral
sbatch ~/run_agloo_qwen.sh
```
The scripts already carry the correct GPU constraint (`--gres=gpu:1 --constraint="vram40|vram48|vram80"`)
and the tool-call parser per model (Qwen hermes, Llama llama3_json, Mistral mistral). Do not use
`--gres=gpu:a100:1` (A100 queue is congested) and do not drop the vram constraint (small GPUs OOM the
14B and even the 8B, and old GPUs fail the CUDA kernel).

## When jobs finish, do this

1. **Over-citation ratio (local, cheap).** For each `(run.json, sidecar.json)` compute the round-1
   and pooled ratio = self-gen share of citations / self-gen share of context. Reuse the counting in
   the T1 compare (`scripts/camera_ready/t1_compare.py`) or the round-1 block in `~/loo_attribution.py`.
   The run JSONs are under `all_experiments/graphite/{agentic_rag,paraphrase,rerank}/...` (local and on
   `/work`). Replace-All and hybrid are ratio 1.0 by construction (context is 100% self-gen at round 1).
2. **R3 (query-alignment).** Submit one CPU job that runs `~/r3_from_sidecar.py` over the new
   paraphrase, rerank, and agentic sidecars, same as job 64015995 did for the baselines. It loads
   all-MiniLM-L6-v2 and prints the self_gen A->B attenuation. It auto-skips variants where self_gen has
   no variation (replace_all, hybrid). Model the job on `~/run_r3_sidecar.sh`.
3. **Fold both into `docs/citation_attribution_spec.md`**, in the "Option B" section, as a cross-model
   agentic + paraphrase + rerank table. Commit and push to `main`.
4. **DeepSeek re-run (64025428).** Output `~/deepseek_agentic_rerun.json`. When done, run
   `~/agentic_loo.py "$HOME/deepseek_agentic_rerun.json" deepseek-r1-distill-qwen-7b <served_api> <out>`
   against a DeepSeek server started with `--reasoning-parser deepseek_r1 --enable-auto-tool-choice
   --tool-call-parser hermes` and `max_tokens 4096`. Then fold its ratio in. This is the only way
   DeepSeek joins the agentic set with real data.

## Caveats to carry into the write-up

- **Agentic LOO is Option A**, the faithful tool-calling flow over the *recorded* store. The store at
  round r is reconstructed from the documents the run recorded through round r (`iteration <= r`), not
  the full growing store (that is not saved). Say this plainly. It is the only tractable faithful
  attribution of the existing run.
- **DeepSeek agentic original run is degenerate**, 84% empty answers, because it was served with
  `max_tokens=512`, which truncated DeepSeek-R1's long reasoning before the final answer. The fix,
  confirmed by smoke (empty rate 92% -> 17%), is `--reasoning-parser deepseek_r1 --tool-call-parser
  hermes --enable-auto-tool-choice` plus `max_tokens=4096`. The re-run (64025428) uses it.
- **R3 Search splits 3/4-null** across models, with DeepSeek the lone positive and lower-confidence
  (reasoning traces inflate the change score). Replace-One is the robust headline (all four models,
  effect grows under controls).
- **The over-citation ratio is partly a prefilter artifact.** The top-2 overlap gate over-selects
  self-gen by ancestry (2-2.9x). Lead the §6 rewrite with the counterfactual and R7, not the raw share.

## Last step for the whole markdown pass

`docs/citation_attribution_spec.md` is the one markdown file not yet deslopped (it was still being
appended to). Once Option B results are folded in and it is stable, deslop it per the poteto writing
standard (remove long dashes and mid-sentence colon connectors, split run-ons), preserving every
number, path, and table. That closes the "deslop every markdown file" task.

## Environment notes

- `ssh unity` works. Activate the env with
  `source /modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1/etc/profile.d/conda.sh; conda activate ragenv`.
- Scripts that import the repo (`agentic_loo.py`) run from `~/RAG-Collapsement-on-Self-Refined-Generation`
  with `export PYTHONPATH=$HOME/RAG-Collapsement-on-Self-Refined-Generation`.
- HF cache: `export HF_HOME=/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/hf_cache`
  with `HF_HUB_OFFLINE=1`.
- The agent cannot push from Unity. Commit and push from the laptop. Never commit API keys.
- Job scripts and outputs are archived in `~/rag_rebuttal_scripts/` on Unity.
