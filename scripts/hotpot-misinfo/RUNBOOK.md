# Runbook — HotpotQA misinformation smoke test on Unity

Goal: validate the error-compounding experiment **end-to-end with real vLLM servers** before
the full pilot. The smoke runs all three arms (faithful / counterfactual / freeform) on 5
questions × 3 rounds and self-checks the output.

> **Quick path (recommended):** `sbatch --export=ALL,MAX_Q=20 scripts/hotpot-misinfo/smoke_all_in_one.sh`
> is a single self-contained GPU job — it starts one Qwen2.5-7B vLLM server (used for both the
> answer and doc-gen/judge roles), waits for it, runs the smoke client + eval + checks against
> `localhost`, and tears the server down on exit. No two-server URL coordination, ~8 min wall,
> 2 h time limit. Use `MAX_Q=20` — a weak 7B answerer needs ~20 questions to reliably land at
> least one realized (`status=ok`) counterfactual injection (5 can all be yes/no or
> model-already-wrong). The two-server flow below is for the representative 14B-answer setup.

## Topology

```
GPU job  server_answer.sh   → Qwen2.5-14B-Instruct  (vLLM, port 5154)   VLLM_API_BASE
GPU job  server_docgen.sh   → Qwen2.5-7B-Instruct   (vLLM, port 5153)   DOC_VLLM_API_BASE
CPU job  smoke_test.sh      → client: faiss-cpu reads the index, E5 on CPU, calls both servers
```
The client is **CPU-only** (it loads the ~7.6 GB FAISS index + corpus into RAM → `--mem=96g`).
Only the two servers need GPUs.

## Files (`scripts/hotpot-misinfo/`)
- `server_answer.sh` — answer-model vLLM server (Qwen2.5-14B; no tool flags, search variant isn't agentic).
- `server_docgen.sh` — doc-generation / judge vLLM server (Qwen2.5-7B).
- `smoke_test.sh` — CPU client: 5 q × 3 rounds × 3 arms + eval + assertions.
- `run_misinfo.sh` — the full 50-question pilot (run after the smoke passes).

## Data artifacts (already on Unity — defaults point here)
- Queries: `/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hf_cache/queries/{test,train,val}`
- FAISS index: `/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpotqa_index/` (`ivf.index`, `docid_map.json`)
- Gold answers: `/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json`

---

## Step 0 — get the code onto Unity

The work is on branch **`misinfo-error-compounding`** (committed locally, not yet on `origin`).
On your laptop:
```bash
git push origin misinfo-error-compounding
```
Then on Unity (in the repo clone), authenticate and pull (HTTPS needs your PAT — see the earlier
note; do this in your own Unity shell):
```bash
cd ~/RAG-Collapsement-on-Self-Refined-Generation
git fetch origin && git checkout misinfo-error-compounding && git pull origin misinfo-error-compounding
mkdir -p logs
python -m pytest tests/test_misinfo.py -q     # 15 tests, ~2s, sanity that the module loads in ragenv
```

## Step 1 — start the two GPU servers

```bash
cd ~/RAG-Collapsement-on-Self-Refined-Generation
sbatch scripts/hotpot-misinfo/server_answer.sh
sbatch scripts/hotpot-misinfo/server_docgen.sh
squeue --me                     # wait until both are RUNNING
```
Each server log prints its URL. Wait until you see `Application startup complete` / `Uvicorn running`
in `logs/slurm-<jobid>-vllm-answer.out` and `…-docgen.out`, then grab the node FQDNs:
```bash
grep -h "export VLLM_API_BASE\|export DOC_VLLM_API_BASE\|Uvicorn running" logs/slurm-*-vllm-*.out
```
Export the two URLs (replace `<host>` with the FQDNs from the logs):
```bash
export VLLM_API_BASE="http://<answer-host>:5154/v1"
export DOC_VLLM_API_BASE="http://<docgen-host>:5153/v1"
```
Sanity-check both are serving (from a login node):
```bash
curl -s "$VLLM_API_BASE/models"      | python -m json.tool   # expect served name "qwen2.5-14b"
curl -s "$DOC_VLLM_API_BASE/models"  | python -m json.tool   # expect "qwen2.5-7b-docgen"
```

## Step 2 — run the smoke (CPU client)

```bash
sbatch --export=ALL scripts/hotpot-misinfo/smoke_test.sh     # forwards the two URLs into the job
squeue --me
tail -f logs/hotpot_misinfo_smoke_*.out
```
Expect ~5–15 min (most of it is loading the corpus + index into RAM the first time).

## Step 3 — success criteria

The job ends with these lines (and a non-zero exit only on failure):
```
PASS parity: faithful arm is byte-compatible (no injection keys)
PASS counterfactual: N questions with status=ok and recorded doc ids
PASS freeform: M eligible; discovery harness populated claims
PASS eval: injection_aggregates present (n_eligible=…, adoption_rate=…); asr_by_iteration=…
ALL SMOKE CHECKS PASSED
```
Spot-check that `asr_by_iteration` is **non-degenerate** — 0.0 at round 0, then rising in later
rounds (the injected doc is created at `--inject-round 1` and enters context from round 2). Outputs
land in `hotpot_misinfo_outputs/smoke/`.

## Step 4 — clean up

```bash
scancel <answer-jobid> <docgen-jobid>      # free the two GPUs once the smoke passes
```

## Step 5 — full pilot (after the smoke passes)

Same two servers, then:
```bash
export GT_FILE=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json
sbatch --export=ALL scripts/hotpot-misinfo/run_misinfo.sh   # 50 q × {faithful,counterfactual,freeform} + eval
```

---

## All 3 variants in parallel, sharing ONE server pair (recommended for the real runs)

The answer and doc-gen servers are **stateless** — a single pair serves all three variant
clients at once (vLLM queues concurrent requests). So you need **2 GPUs for servers, not 6**.
One command does everything:

```bash
export GT_FILE=/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/hotpot_dev_fullwiki_v1.json
bash scripts/hotpot-misinfo/launch_all_variants.sh      # run on a LOGIN node
```

It submits the 2 GPU servers, waits until both serve (parses each server log for its URL +
curls `/models`), fans out one **CPU** client per variant (`search` / `replace_one` /
`hybrid`, each running all 3 arms) pointed at the shared URLs, and submits a **reaper**
(`--dependency=afterany` on the clients) that `scancel`s the servers when every variant
finishes. Total footprint: **2 GPUs + 3 CPU jobs**.

Time limits respect a **2-day job cap**: servers request 48h (`SERVER_TIME`) but the reaper
kills them early; clients request **47h** (`CLIENT_TIME`) so the servers — which start
~10-15 min earlier — always outlive them. Each arm's output is written before the next arm
starts, so a wall-clock kill only loses the in-flight arm.

Useful overrides (env): `VARIANTS="search replace_one"`, `MAX_Q`, `SEED`, `TARGET`,
`ARMS="counterfactual freeform"`, `CLIENT_TIME`, `SERVER_TIME`. If one variant's 3-arm run
ever risks the 47h wall at your scale, split it: launch per-arm jobs with `ARMS=<one arm>`
(all still share the same server pair).

Monitor with `squeue --me` and `tail -f logs/hotpot_misinfo_cli_*.out`; cancel everything
with the `scancel` line the launcher prints.

---

## Troubleshooting

- **Server "model not found" / name mismatch** — the client passes `--model-name qwen2.5-14b`
  / `qwen2.5-7b-docgen`; these must equal each server's `--served-model-name` (the curl in Step 1
  confirms). Override with `MODEL=` / `DOC_MODEL=` env vars if you change a served name.
- **Servers not reachable** — Unity compute nodes resolve each other by FQDN; make sure you used
  the FQDN from the log (`hostname -f`), not `localhost`. Re-run the `curl` check.
- **`gt-file` permission denied** — the native dev JSON is under oyilmazel's scratch; if you can't
  read it, copy it into your own scratch and point `GT_FILE` there.
- **OOM on the client** — raise `--mem` (corpus + index are large); the client is CPU so use the
  `cpu` partition (already set).
- **No eligible injections in the smoke** — 5 questions may all be yes/no or model-already-wrong;
  raise `--max-questions` (edit `smoke_test.sh`) or check the `statuses=` line it prints.
- **`_id` alignment** — gold is looked up by the mteb `query_id`; if many questions report
  `skipped_no_gold`, the `--split` (default `test`) and the GT file are mismatched. The dev
  fullwiki file aligns with the `test` split queries used here.
- **faiss / E5 slow** — first load reads ~7.6 GB from scratch; subsequent steps are fast. This is
  inherent to the hotpot pipeline, not the misinfo code.
