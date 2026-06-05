# Unity HPC Setup Guide

## Storage overview (as of 2026-06-03)

| Filesystem | Used | Quota | % |
|---|---|---|---|
| `/home/rsenapati_umass_edu` | 58.80 GB | 107.37 GB | 55% |
| `/project/pi_dagarwal_umass_edu` | 16.10 TB | 16.49 TB | 98% ⚠️ near full |
| `/work/pi_dagarwal_umass_edu` | 2.37 TB | 3.22 TB | 74% |
| `/scratch4/workspace/rsenapati_umass_edu-rag-collapse` | — | 15 TB | active |

**Key paths:**
- Experiment outputs: `/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu/experiment_outputs/`
- HuggingFace model cache: `/scratch4/workspace/rsenapati_umass_edu-rag-collapse/hf_cache`

---

## Scratch space (HPC Workspace)

Unity uses the `ws_*` tool to manage scratch workspaces. Scratch has no snapshots — do not store the only copy of important files there. New workspaces are auto-assigned to whichever scratch filesystem (`scratch`, `scratch3`, `scratch4`) has the most free space.

### Create the workspace (done — expires 2026-07-03)

```bash
ws_allocate -m rsenapati@umass.edu -r 3 rag-collapse 30
# Created: /scratch4/workspace/rsenapati_umass_edu-rag-collapse
```

Shared with the PI group:
```bash
ws_allocate -G pi_dagarwal_umass_edu rag-collapse 30
```

### Create the HuggingFace cache subdirectory

```bash
mkdir -p /scratch4/workspace/rsenapati_umass_edu-rag-collapse/hf_cache
```

All vLLM server scripts set `HF_HOME` and `HF_HUB_CACHE` to this path so model weights are cached here and not re-downloaded between jobs.

### Useful commands

```bash
ws_list -v                    # list workspaces with expiry and path
ws_extend rag-collapse 30     # extend by up to 30 days (10,000 extensions available)
squota                        # check quota usage across all scratch filesystems
ws_release rag-collapse       # release when done (frees the ID; deletion happens later)
```

### Email reminder

A reminder is sent to `rsenapati@umass.edu` 3 days before expiry (2026-06-30). To change the default:

```bash
cat ~/.ws_user.conf
# reminder: 3
# mail: rsenapati@umass.edu
```

### Share with a labmate

```bash
ws_share share rag-collapse <other_username>
ws_share list rag-collapse        # see who has access
ws_share unshare rag-collapse <other_username>
```

---

## Running the GEPA pipeline jobs

### Step 1 — Start servers (5 jobs)

```bash
sbatch scripts/gepa_pipeline/server_qwen7b_docgen.sh   # shared doc-gen, port 5153
sbatch scripts/gepa_pipeline/server_mistral7b.sh        # port 5151
sbatch scripts/gepa_pipeline/server_deepseek_r1_7b.sh  # port 5152
sbatch scripts/gepa_pipeline/server_qwen14b.sh          # port 5154
sbatch scripts/gepa_pipeline/server_llama3.1_8b.sh     # port 5155
```

Wait until each prints `Uvicorn running on http://0.0.0.0:<port>/v1`, then note the FQDN from the log:
```
Reachable at: http://gpu032.unity.rc.umass.edu:5151/v1
```

### Step 2 — Edit and submit pipeline jobs

Fill in the two URL placeholders at the top of each run script, then submit:

```bash
# Edit VLLM_API_BASE and DOC_VLLM_API_BASE, then:
sbatch scripts/gepa_pipeline/run_mistral7b.sh
sbatch scripts/gepa_pipeline/run_deepseek_r1_7b.sh
sbatch scripts/gepa_pipeline/run_qwen14b.sh
sbatch scripts/gepa_pipeline/run_llama3.1_8b.sh
```

Each run script executes replace_all → replace_one → search sequentially with `--use-gepa-prompt`. Walltime is 4 days.

### Step 3 — Monitor

```bash
squeue -u rsenapati_umass_edu
tail -f logs/gepa_mistral7b_<jobid>.out
```

### Step 4 — Parse results after a GEPA optimization run

```bash
python gepa_optimization/parse_results.py \
  --run-dir gepa_runs/<run_name> \
  --min-evals 10
```

Apply the best prompt to `formatters.py`:
```bash
python gepa_optimization/apply_best_prompt.py --prompt "..."
```
