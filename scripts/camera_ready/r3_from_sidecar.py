"""R3 query-alignment control, reading citations from an Option-B sidecar instead of the run JSON.
Same model as r3_qalign.py: y = per-doc citation rate; covariates self_gen, qcos (MiniLM cos of
question,doc), redund, position, loglen; CR0 (question-clustered) OLS; report self_gen A->B attenuation.

Usage: python r3_from_sidecar.py <run.json>|<sidecar.json>|<label> [more ...]
Sidecar shape: {qi: {round: {run_id: [doc_id, ...]}}}.
"""
import json, re, sys, math
import numpy as np
from sentence_transformers import SentenceTransformer

PAIRS = [a.split("|") for a in sys.argv[1:]]
WORD = re.compile(r"[A-Za-z0-9']+")
ROUNDS = list(range(1, 30))

def tok(t):
    return {w.lower() for w in WORD.findall(t or "") if len(w) >= 4}

def is_ai(doc_id, url, it):
    doc_id = str(doc_id or "").lower(); url = str(url or "").lower()
    return doc_id.startswith("gen_") or url == "model_generated" or (isinstance(it, int) and it > 0 and not doc_id.startswith("ref_"))

print("loading all-MiniLM-L6-v2 ...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

def run_one(run_path, sidecar_path, label):
    d = json.load(open(run_path)); side = json.load(open(sidecar_path))
    rows = []; texts = {}
    def tid(t):
        if t not in texts: texts[t] = len(texts)
        return texts[t]
    for qi, q in enumerate(d["questions"]):
        qidx = tid("Q::" + (q.get("question_text", "") or ""))
        its = {it["iteration_number"]: it for it in q["iterations"]}
        for r in ROUNDS:
            it = its.get(r)
            if not it: continue
            docs = it.get("documents") or []
            runs = it.get("runs") or []
            n_runs = len(runs) or 1
            # citation counts per doc_id this round, from the sidecar
            srr = side.get(str(qi), {}).get(str(r), {})
            cnt = {}
            for run in runs:
                for did in srr.get(run["run_id"], []):
                    cnt[did] = cnt.get(did, 0) + 1
            # de-dup docs by doc_id (agentic repeats)
            seen = {}
            for dc in docs:
                did = dc.get("doc_id")
                if did and did not in seen: seen[did] = dc
            docl = list(seen.values()); ndoc = len(docl)
            for pos, dc in enumerate(docl):
                did = dc.get("doc_id"); txt = dc.get("text", "") or ""
                if not did: continue
                rows.append(dict(q=qi, qidx=qidx, didx=tid("D::" + txt),
                                 y=cnt.get(did, 0) / n_runs,
                                 self=1.0 if is_ai(did, dc.get("url"), dc.get("iteration")) else 0.0,
                                 pos=pos / max(1, ndoc - 1), loglen=math.log(1 + len(txt)),
                                 ctx=[tid("D::" + (o.get("text", "") or "")) for o in docl if o.get("doc_id") != did]))
    if not rows:
        print(f"=== R3 {label}: no rows ==="); return
    if len({r["self"] for r in rows}) < 2:
        print(f"=== R3 {label}: self_gen has no variation (context all self-gen, e.g. replace_all) — R3 not applicable ===\n", flush=True)
        return
    inv = [None] * len(texts)
    for t, i in texts.items(): inv[i] = t[3:]
    emb = np.asarray(model.encode(inv, batch_size=256, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    for row in rows:
        qv = emb[row["qidx"]]; dv = emb[row["didx"]]
        row["qcos"] = float(qv @ dv)
        row["redund"] = float(np.mean(emb[row["ctx"]] @ dv)) if row["ctx"] else 0.0
    y = np.array([r["y"] for r in rows]); qid = np.array([r["q"] for r in rows])
    def ols(cols):
        X = np.column_stack([np.ones(len(rows))] + [np.array([r[c] for r in rows]) for c in cols])
        XtXi = np.linalg.inv(X.T @ X); beta = XtXi @ (X.T @ y); e = y - X @ beta
        meat = np.zeros((X.shape[1], X.shape[1]))
        for g in np.unique(qid):
            Xg = X[qid == g]; s = Xg.T @ e[qid == g]; meat += np.outer(s, s)
        se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
        return dict(zip(["const"] + cols, beta)), dict(zip(["const"] + cols, se))
    bA, sA = ols(["self"]); bB, sB = ols(["self", "qcos", "redund", "pos", "loglen"])
    att = 100 * (1 - bB["self"] / bA["self"]) if bA["self"] else float("nan")
    print(f"=== R3 {label} (rows={len(rows)}) ===")
    print(f"   A self_gen = {bA['self']:+.4f} (t={bA['self']/sA['self']:.1f})")
    print(f"   B self_gen = {bB['self']:+.4f} (t={bB['self']/sB['self']:.1f}) | qcos {bB['qcos']:+.3f} redund {bB['redund']:+.3f}")
    print(f"   attenuation {att:.0f}%\n", flush=True)

for parts in PAIRS:
    run_one(parts[0], parts[1], parts[2] if len(parts) > 2 else parts[0])
print("### r3_from_sidecar done ###", flush=True)
