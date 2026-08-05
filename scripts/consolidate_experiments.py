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
# oyilmazel's GRAPHITE baselines are stale (superseded by ffatima/ratirastogi).
# His HotpotQA data is the sole copy and must be kept.
DROP_GRAPHITE_BASELINE_OWNERS = {"oyilmazel"}

# Scratch dirs pulled in for RERANK data only (gap-fill — reranker eval/entity
# for the non-Qwen models and the oracle/desklib/finetuned ablations live here,
# not in /work). A scratch file is added only if its (dataset, model, filename)
# isn't already covered by /work.
SCRATCH_ROOT = "/work/pi_dagarwal_umass_edu/project_4/file_storage/rsenapati_umass_edu"
SCRATCH_RERANK_SOURCES = [
    (os.path.join(SCRATCH_ROOT, "oyilmazel_umass_edu"), "oyilmazel"),
    (os.path.join(SCRATCH_ROOT, "ffatima_umass_edu"), "ffatima"),
]


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


def classify(src: str, root_dir: str, short: str):
    """Map a source file to its destination tuple, or None if filtered out.
    Layout-agnostic: the output tree may appear at any depth, and the dataset
    dir ('hotpotqa') may sit before OR after it (/work uses <tree>/hotpotqa/...,
    scratch uses hotpotqa/<tree>/...).
    entry = (src, owner_short, dataset, method, cfg, tree, rel, dest_no_owner)"""
    if kept_model(src) is None:
        return None
    relfull = os.path.relpath(src, root_dir)
    parts = relfull.split(os.sep)
    tree = next((p for p in parts if p in TREES), None)
    if tree is None:
        return None  # not an output-tree file
    ti = parts.index(tree)
    ds = "hotpotqa" if "hotpot" in relfull.lower() else "graphite"
    # rel = everything except the tree component and any redundant 'hotpotqa' dir
    rel_parts = [p for i, p in enumerate(parts) if i != ti and p != "hotpotqa"]
    rel = os.path.join(*rel_parts)
    method, cfg = METHOD_MAP[variant_of(os.path.basename(src))]
    if short == DROP_OWNER_NONAGENTIC and method != "agentic_rag":
        return None
    if ds == "graphite" and method == "baseline" and short in DROP_GRAPHITE_BASELINE_OWNERS:
        return None
    if ds == "graphite" and not graphite_keep(src):
        return None
    dst_parts = [ds, method] + ([cfg] if cfg else []) + [tree]
    dest_no_owner = os.path.join(DEST_ROOT, *dst_parts, rel)
    return (src, short, ds, method, cfg, tree, rel, dest_no_owner)


def main() -> int:
    if not os.path.isdir(WORK):
        print(f"ERROR: source root not found: {WORK}", file=sys.stderr)
        return 1
    os.makedirs(DEST_ROOT, exist_ok=True)

    # ---- Pass 1: enumerate every file and its owner-less destination ----
    # entry = (src, owner_short, dataset, method, cfg, tree, rel, dest_no_owner)
    entries = []
    seen_keys = set()  # (dataset, model, filename) — dedups /work vs scratch and scratch vs scratch

    def walk_json(root_dir):
        if not os.path.isdir(root_dir):
            return
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d != "hf_cache"]
            for fn in filenames:
                if fn.endswith(".json"):
                    yield os.path.join(dirpath, fn)

    # Pass 1a — /work (all methods)
    for owner in OWNERS:
        short = owner.replace("_umass_edu", "")
        root_dir = os.path.join(WORK, owner)
        for src in walk_json(root_dir):
            e = classify(src, root_dir, short)
            if e is None:
                continue
            entries.append(e)
            seen_keys.add((e[2], kept_model(src), e[5], os.path.basename(src)))

    # Pass 1b — scratch (RERANK only, gap-fill: skip anything already covered)
    for root_dir, short in SCRATCH_RERANK_SOURCES:
        for src in walk_json(root_dir):
            e = classify(src, root_dir, short)
            if e is None or e[3] != "rerank":
                continue
            key = (e[2], kept_model(src), e[5], os.path.basename(src))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            entries.append(e)

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
