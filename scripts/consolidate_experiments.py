#!/usr/bin/env python3
"""Consolidate all RAG-collapse experiment JSON outputs into a single tree.

Source: /work/pi_dagarwal_umass_edu/project_4/file_storage/<owner>/<output_tree>/...
Dest:   /work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments/
            <dataset>/<method>[/<config>]/<output_tree>[/<owner>]/<rel>

  dataset      = graphite | hotpotqa     (hotpotqa if 'hotpot' appears in the rel path)
  method[/cfg] = baseline/{replace_all,replace_one,search}
               | paraphrase/{search,replace_one,hybrid}
               | agentic_rag | rerank | misc
  output_tree  = entity_extraction_output | evaluation_outputs | experiment_outputs
  owner        = ffatima | ratirastogi | rsenapati | oyilmazel | reranker
                 -- inserted ONLY when >1 owner produced the same file
                    (so duplicate runs are kept and disambiguated; unique runs are flat)

Files are COPIED (originals untouched). Idempotent: skips a file if the dest
already exists with the same size.
"""
import csv
import json
import os
import shutil
import sys
from collections import defaultdict

# Full Graphite benchmark size. Graphite files with a different question count
# (50-question runs, smoke/debug runs) are dropped.
GRAPHITE_N_QUESTIONS = 400

WORK = "/work/pi_dagarwal_umass_edu/project_4/file_storage"
DEST_ROOT = os.path.join(WORK, "all_experiments")
OWNERS = [
    "ffatima_umass_edu",
    "ratirastogi_umass_edu",
    "rsenapati_umass_edu",
    "oyilmazel_umass_edu",
    "reranker",
]
TREES = ["entity_extraction_output", "evaluation_outputs", "experiment_outputs"]

# Only these 4 models are kept (the gepa_pipeline set, excluding the qwen2.5-7b docgen).
KEEP_MODELS = {
    "Qwen2.5-14B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
}
# rsenapati's runs are dropped EXCEPT agentic_rag (his focus area).
DROP_OWNER_NONAGENTIC = "rsenapati"


def kept_model(path: str):
    """Return the KEEP_MODELS name found in the path, else None (drop the file)."""
    q = path.replace("__", "/")
    for m in KEEP_MODELS:
        if m in q:
            return m
    return None


def variant_of(fname: str) -> str:
    n = fname.lower()
    if "hybrid_paraphrased" in n:
        return "hybrid_paraphrase"
    if "replace_one_paraphrased" in n:
        return "replace_one_paraphrase"
    if "search_paraphrased" in n:
        return "search_paraphrase"
    if "rerank" in n:
        return "rerank"
    if "agentic_rag" in n:
        return "agentic_rag"
    if "replace_all" in n:
        return "replace_all"
    if "replace_one" in n:
        return "replace_one"
    if "search" in n:
        return "search"
    return "misc"


# variant -> (method, config-or-None) path segments
METHOD_MAP = {
    "replace_all": ("baseline", "replace_all"),
    "replace_one": ("baseline", "replace_one"),
    "search": ("baseline", "search"),
    "search_paraphrase": ("paraphrase", "search"),
    "replace_one_paraphrase": ("paraphrase", "replace_one"),
    "hybrid_paraphrase": ("paraphrase", "hybrid"),
    "agentic_rag": ("agentic_rag", None),
    "rerank": ("rerank", None),
    "misc": ("misc", None),
}


def dataset_of(rel_path: str) -> str:
    return "hotpotqa" if "hotpot" in rel_path.lower() else "graphite"


def graphite_keep(path: str) -> bool:
    """Keep a graphite file only if it holds the full 400-question benchmark.
    On a read/permission error or unknown schema, keep it (let the copy step
    decide / record), so we never silently drop a real run."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (PermissionError, OSError):
        return True  # unreadable -> let copy step log it in UNREADABLE.txt
    except Exception:
        return True  # malformed -> don't silently drop
    q = d.get("questions") if isinstance(d, dict) else None
    if isinstance(q, list):
        return len(q) == GRAPHITE_N_QUESTIONS
    return True  # unknown schema -> keep


def main() -> int:
    if not os.path.isdir(WORK):
        print(f"ERROR: source root not found: {WORK}", file=sys.stderr)
        return 1
    os.makedirs(DEST_ROOT, exist_ok=True)

    # ---- Pass 1: enumerate every file and its owner-less destination ----
    # entry = (src, owner_short, dataset, method, cfg, tree, rel, dest_no_owner)
    entries = []
    for owner in OWNERS:
        short = owner.replace("_umass_edu", "")
        for tree in TREES:
            src_root = os.path.join(WORK, owner, tree)
            if not os.path.isdir(src_root):
                continue
            for dirpath, dirnames, filenames in os.walk(src_root):
                dirnames[:] = [d for d in dirnames if d != "hf_cache"]
                for fn in filenames:
                    if not fn.endswith(".json"):
                        continue
                    src = os.path.join(dirpath, fn)
                    # filter: keep only the 4 target models
                    if kept_model(src) is None:
                        continue
                    rel = os.path.relpath(src, src_root)
                    ds = dataset_of(rel)
                    # drop a redundant leading 'hotpotqa/' (oyilmazel nests hotpot
                    # under evaluation_outputs/hotpotqa/...) — dataset is already the top folder
                    relparts = rel.split(os.sep)
                    if ds == "hotpotqa" and len(relparts) > 1 and relparts[0] == "hotpotqa":
                        rel = os.path.join(*relparts[1:])
                    method, cfg = METHOD_MAP[variant_of(fn)]
                    # filter: drop rsenapati's non-agentic runs
                    if short == DROP_OWNER_NONAGENTIC and method != "agentic_rag":
                        continue
                    # filter: graphite must be the full 400-question benchmark
                    if ds == "graphite" and not graphite_keep(src):
                        continue
                    parts = [ds, method] + ([cfg] if cfg else []) + [tree]
                    dest_no_owner = os.path.join(DEST_ROOT, *parts, rel)
                    entries.append((src, short, ds, method, cfg, tree, rel, dest_no_owner))

    # ---- group by owner-less dest: a collision means >1 owner ran it ----
    groups = defaultdict(list)
    for e in entries:
        groups[e[7]].append(e)

    # ---- Pass 2: copy ----
    manifest, unreadable = [], []
    copied = skipped = 0
    counts = defaultdict(int)

    for dno, items in groups.items():
        multi = len(items) > 1
        for (src, short, ds, method, cfg, tree, rel, dest_no_owner) in items:
            if multi:
                # insert <owner> between the <output_tree> dir and the relative path
                base = dest_no_owner[: len(dest_no_owner) - len(rel)].rstrip(os.sep)
                dest = os.path.join(base, short, rel)
            else:
                dest = dest_no_owner

            label = method + ("/" + cfg if cfg else "")
            counts[(ds, label, tree)] += 1
            manifest.append((src, dest, ds, label, tree, short, "dup" if multi else "unique"))

            if os.path.exists(dest) and os.path.getsize(dest) == os.path.getsize(src):
                skipped += 1
                continue
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
            except (PermissionError, OSError) as e:
                unreadable.append((src, str(e)))
                continue
            copied += 1
            if copied % 25 == 0:
                print(f"  ... {copied} copied", flush=True)

    man_path = os.path.join(DEST_ROOT, "MANIFEST.csv")
    with open(man_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dest", "dataset", "method", "output_tree", "owner", "kind"])
        w.writerows(manifest)

    print("\n===== SUMMARY =====")
    print(f"copied={copied}  skipped(existing)={skipped}  "
          f"unreadable={len(unreadable)}  total={len(manifest)}")
    print(f"manifest: {man_path}")
    if unreadable:
        ur_path = os.path.join(DEST_ROOT, "UNREADABLE.txt")
        with open(ur_path, "w") as f:
            for src, err in unreadable:
                f.write(f"{src}\t{err}\n")
        print(f"unreadable list: {ur_path}  (source files we lack read permission on)")
    print()
    for (ds, label, tree), n in sorted(counts.items()):
        print(f"  {ds:9s} {label:24s} {tree:25s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
