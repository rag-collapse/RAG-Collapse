"""Does the top-M overlap prefilter inflate the over-citation ratio?

Only the top citation_top_m=2 documents by lexical overlap with an answer are eligible to be
cited (pipeline.py:76). Self-generated docs share wording with the answer by ancestry, so they
may be over-represented in that eligible set independent of influence. If they are, the prefilter
itself lifts the self-gen cited share and inflates the §6.2 ratio.

This measures it directly from the reported Qwen-14B Replace-One run, replicating the pipeline's
own token set (words >= 4 chars, lowercased) and overlap score (|answer & doc| / |answer|). For
each (question, round, run answer) it ranks the context docs, takes the top 2, and compares the
self-gen share of that eligible set to the self-gen share of the whole context.

  amplification ratio = selfgen_share(top-2 eligible) / selfgen_share(context)

Ratio ~ 1 means the gate is provenance-neutral and does not threaten §6.2. Ratio > 1 bounds the
inflation the prefilter contributes on its own, before any cite decision.
"""
import json, re, os
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SRC = os.path.join(ROOT, "all_experiments", "graphite", "baseline", "replace_one",
                   "experiment_outputs", "Qwen", "Qwen2.5-14B-Instruct", "local_replace_one.json")
TOP_M = 2
WORD = re.compile(r"[A-Za-z0-9']+")

def toks(t):
    return {w.lower() for w in WORD.findall(t or "") if len(w) >= 4}

def overlap(a, d):
    return len(a & d) / max(1, len(a)) if a else 0.0

def is_ai(doc_id, url, it):
    doc_id = str(doc_id or "").lower(); url = str(url or "").lower()
    return doc_id.startswith("gen_") or url == "model_generated" or (isinstance(it, int) and it > 0 and not doc_id.startswith("ref_"))

d = json.load(open(SRC))
# per round: [selfgen in top-2 eligible], [total top-2], [selfgen in context], [total context]
top_sg = defaultdict(int); top_n = defaultdict(int)
ctx_sg = defaultdict(int); ctx_n = defaultdict(int)

for q in d["questions"]:
    for it in q["iterations"]:
        r = it["iteration_number"]
        if r == 0:
            continue
        docs = it.get("documents") or []
        runs = it.get("runs") or []
        if not docs or not runs:
            continue
        doc_tok = [(dc, toks(dc.get("text", "")),
                    is_ai(dc.get("doc_id"), dc.get("url"), dc.get("iteration"))) for dc in docs]
        # context self-gen share counted once per (round) using the doc set, per answer for eligibility
        for run in runs:
            at = toks(run.get("answer", ""))
            scored = sorted(doc_tok, key=lambda x: overlap(at, x[1]), reverse=True)
            elig = scored[:max(1, TOP_M)]
            for _, _, sg in elig:
                top_n[r] += 1; top_sg[r] += int(sg)
            for _, _, sg in doc_tok:
                ctx_n[r] += 1; ctx_sg[r] += int(sg)

print(f"top_m={TOP_M}")
print(f"{'round':>5} {'ctx_sg%':>8} {'topM_sg%':>9} {'amplif':>7}")
for r in sorted(ctx_n):
    if r > 12 and r not in (15, 19):
        continue
    cs = ctx_sg[r] / ctx_n[r]; ts = top_sg[r] / top_n[r]
    amp = ts / cs if cs else float("nan")
    print(f"{r:5d} {100*cs:8.1f} {100*ts:9.1f} {amp:7.2f}")

# gate-width sensitivity at round 1: wider gate -> eligibility approaches context share
print("\nround-1 amplification vs gate width top_m:")
r = 1
for M in (1, 2, 4, 8, 999):
    sg = n = csg = cn = 0
    for q in d["questions"]:
        for it in q["iterations"]:
            if it["iteration_number"] != r:
                continue
            docs = it.get("documents") or []
            dt = [(toks(dc.get("text", "")), is_ai(dc.get("doc_id"), dc.get("url"), dc.get("iteration"))) for dc in docs]
            for run in (it.get("runs") or []):
                at = toks(run.get("answer", ""))
                elig = sorted(dt, key=lambda x: overlap(at, x[0]), reverse=True)[:max(1, M)]
                for _, s in elig:
                    n += 1; sg += int(s)
                for _, s in dt:
                    cn += 1; csg += int(s)
    label = "all" if M == 999 else str(M)
    print(f"  top_m={label:>3}: ctx {100*csg/cn:.1f}%  eligible {100*sg/n:.1f}%  amplif {(sg/n)/(csg/cn):.2f}")
