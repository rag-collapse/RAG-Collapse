"""R7 (token-level): novelty-restriction influence measure. Overlap-free, so it is immune to the
ancestry/prefilter confound that inflates the citation-share ratio.

Chain: A_{k-1} -> D=gen_k_0 -> A_k. The document generator expands the round-(k-1) answer into a
synthetic article and, doing so, invents material that was not in that answer. Call the invented
tokens novel(D):

  novel(D) = tokens(D) minus tokens(all round-(k-1) answers) minus tokens(other round-k context
             docs) minus tokens(question)

Under pure ancestry A_k has no route to novel(D): it is not in the prior answer, not in the other
context, not in the question. The document is the only carrier. So the adoption of novel(D) by the
next answer A_k is influence that cannot be ancestry. A cross-question placebo (the same novel set
scored against a different question's A_k, where D was never in context) gives the null rate at
which these tokens appear anyway.

Reports, per round and pooled: same-question adoption, cross-question placebo, and the difference
(the influence attributable to the document being present). No generation; runs on the stored run.
"""
import json, re, os, random
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SRC = os.path.join(ROOT, "all_experiments", "graphite", "baseline", "replace_one",
                   "experiment_outputs", "Qwen", "Qwen2.5-14B-Instruct", "local_replace_one.json")
WORD = re.compile(r"[A-Za-z0-9']+")
random.seed(13)

def tok(t):
    return {w.lower() for w in WORD.findall(t or "") if len(w) >= 4}

d = json.load(open(SRC))
Q = d["questions"]

# precompute per question: its[round] -> dict, answers-union per round, question tokens
qdata = []
for q in Q:
    its = {it["iteration_number"]: it for it in q["iterations"]}
    ans_union = {r: set().union(*[tok(run["answer"]) for run in its[r]["runs"]]) if its[r].get("runs") else set()
                 for r in its}
    ans_runs = {r: [tok(run["answer"]) for run in (its[r].get("runs") or [])] for r in its}
    qdata.append(dict(its=its, ans_union=ans_union, ans_runs=ans_runs, qtok=tok(q.get("question_text", ""))))

def novel_set(qi, k):
    """novel tokens D=gen_k_0 introduced at round k for question qi."""
    q = qdata[qi]; its = q["its"]
    if k not in its or (k - 1) not in q["ans_union"]:
        return None
    round_docs = its[k].get("documents") or []
    dtok = None; other = set()
    for x in round_docs:
        did = x.get("doc_id")
        t = tok(x.get("text"))
        if did == f"gen_{k}_0":
            dtok = t
        else:
            other |= t
    if not dtok:
        return None
    novel = dtok - q["ans_union"][k - 1] - other - q["qtok"]
    return novel

def adoption(novel, runs_tok):
    if not novel or not runs_tok:
        return None
    return sum(len(novel & rt) / len(novel) for rt in runs_tok) / len(runs_tok)

by_round = defaultdict(lambda: {"same": [], "cross": [], "nsize": []})
n_q = len(qdata)
for qi in range(n_q):
    for k in range(1, 20):
        novel = novel_set(qi, k)
        if novel is None or len(novel) == 0:
            continue
        same = adoption(novel, qdata[qi]["ans_runs"].get(k))
        if same is None:
            continue
        # cross-question placebo: a different question's round-k answers
        qj = qi
        for _ in range(8):
            qj = random.randrange(n_q)
            if qj != qi and qdata[qj]["ans_runs"].get(k):
                break
        cross = adoption(novel, qdata[qj]["ans_runs"].get(k))
        by_round[k]["same"].append(same)
        if cross is not None:
            by_round[k]["cross"].append(cross)
        by_round[k]["nsize"].append(len(novel))

def m(v):
    return sum(v) / len(v) if v else float("nan")

print(f"{'round':>5} {'n_pairs':>7} {'novel|D|':>8} {'same-Q adopt':>13} {'cross-Q adopt':>14} {'influence(d)':>13}")
allsame, allcross = [], []
for k in sorted(by_round):
    b = by_round[k]
    s = m(b["same"]); c = m(b["cross"])
    allsame += b["same"]; allcross += b["cross"]
    print(f"{k:5d} {len(b['same']):7d} {m(b['nsize']):8.1f} {s:13.3f} {c:14.3f} {s-c:+13.3f}")
print("-" * 66)
print(f"{'POOL':>5} {len(allsame):7d} {'':8} {m(allsame):13.3f} {m(allcross):14.3f} {m(allsame)-m(allcross):+13.3f}")
print("\nsame-Q adopt = fraction of D's invented tokens the NEXT answer picks up (mean over 10 runs).")
print("cross-Q adopt = same novel set vs a different question's round-k answers (placebo/null).")
print("influence = same - cross: adoption that requires D to have been in context (not ancestry).")
