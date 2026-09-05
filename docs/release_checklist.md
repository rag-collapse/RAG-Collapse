# R6: Repo release checklist (camera-ready)

NbXB counted "planned for release" against the paper. hAN7 explicitly did **not** count the
datasets against it because they were disclosed. Releasing removes the cheapest objection on
the list. This tracks exactly what is ready, what blocks a public release, and the audit that
was run. Status as of the R4/R5 pass (branch `citation-attribution-spec`).

## Ready to release (already committed)

| Item | Where | Notes |
|---|---|---|
| Pipeline + all method variants | `pipeline.py`, `pipeline/`, `hotpot_pipeline.py`, rerank pipelines | full collapse loop |
| LLM backends | `llm_service/` | vLLM / LiteLLM / server; keys via `os.environ` only |
| GEPA subsystem | `gepa_optimization/`, `gepa_runs/` | prompt optimization |
| **Entity dataset** | `datasets/umass_data.entity.chatgpt.{50,400}.jsonl` | committed (69 MB) |
| **Source-level citation labels / direct-elicitation data** | `evaluation_outputs/Qwen/citations/*.json` | the explicit-elicitation ("direct") citation eval outputs |
| Metrics code | `evaluation.py`, `entity_extraction.py`, `hotpot_evaluation.py` | |
| Reviewer reanalysis (this rebuttal) | `docs/citation_attribution_spec.md`, `scripts/camera_ready/` | R1–R5 + R3b |
| Reproduction docs | `README.md`, `docs/`, `CLAUDE.md` | full workflow |

## Blockers (must resolve before flipping public)

1. **No LICENSE file** (`gh repo view` reports `licenseInfo: null`). A code+data release needs an
   explicit license. **Blocking.** Decision required (see "Open decisions").
2. **Repo is PRIVATE** (`rag-collapse/RAG-Collapsement-on-Self-Refined-Generation`). Going
   public exposes the **entire git history**, not just the current tree. **Blocking**, and
   see the history audit below before flipping.

## Pre-release audit (run this pass)

- **Secrets, current tree:** CLEAN. No hardcoded keys/tokens in any tracked `.py/.sh/.md`.
  All credentials read from `os.environ` / `getenv`. The keymaker key is env-only.
- **Secrets, full history:** CLEAN. The scan was restricted to code/text paths across *all* commits
  (`git log --all -G` for `sk-…`, `AKIA…`, `ghp_…`, `AIza…`, and a keymaker-key literal) and found
  **0 matches**. No secret-shaped literal was ever committed to a tracked code/text file. (The
  scan skips large data/PNG blobs for speed; those are dataset text and plots, not credentials.)
- **Determinism / "seeds":** generation runs at `temperature=0.7` (sampling), and no RNG seed is
  set, so runs are **not bitwise-reproducible**. Reproducibility rests on (a) the released
  experiment/eval outputs and (b) the 10-runs-per-round design. State this honestly in the
  README rather than implying exact-string reproduction. If a reviewer wants determinism, add
  `seed=` and `temperature=0` to the sampling params and document it.
- **Third-party data paths:** code and docs reference Unity `/work/...` paths and a collaborator's
  outputs (Ozel's HotpotQA runs). These are config strings, not data, so they are fine to publish. The
  large raw experiment outputs live on Unity, not in the repo.

## Decisions (made)

1. **License, DECIDED: MIT (code) + CC-BY-4.0 (data).** Added: `LICENSE` (MIT),
   `DATA_LICENSE.md` (CC-BY-4.0), `CITATION.cff`, and a License section in `README.md`.
2. **Release mechanism, DEFERRED (team).** Visibility is left PRIVATE for now, and everything else is
   one step from ready. When the team decides, either flip this repo public (which publishes the full
   history, clean per the scan but heavy: ~356 viz PNGs, `gepa_runs/`, `1400q_summaries_old/`)
   or push a clean squashed mirror to curate what ships.

## Remaining before public (team actions, not automated here)

1. **Fill `CITATION.cff` authors.** The file ships with a `TODO` placeholder. Replace it with the
   full paper author list in order (do not publish the placeholder).
2. **Confirm upstream data terms.** The entity dataset derives from an internal `umass_data`
   source. Verify redistribution under CC-BY-4.0 is permitted and add the upstream attribution
   in `DATA_LICENSE.md` (or ship a regeneration script instead of the JSONL).
3. **Flip visibility** (or push the mirror). Team action.
4. **Tag a release** (e.g. `v1.0-camera-ready`) so reviewers can cite a fixed commit.
