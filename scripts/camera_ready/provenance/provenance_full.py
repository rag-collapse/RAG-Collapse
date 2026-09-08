"""
Full-scale (400-question) three-way citation analysis from the all_experiments
Qwen2.5-14B raw JSON, for Replace-One and Search. LLM quality scores cover only
the round-1 subset (see provenance_regression.py), so here we control for cheap
proxies (doc length, position in context, lexical relevance to the question).

Provenance:
  answer_derived : doc_id starts with "gen_" (model's own prior answer, url=model_generated)
  ai_original    : original ref, detector says AI/Mixed
  human_original : original ref, detector says Human

citation_rate(doc) = citations to that doc in the round / total citations in the round
(aggregated over the 10 runs).
"""

import csv
import json
import os
import re
import sys
from urllib.parse import urlsplit

import numpy as np

DETECTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_detector.csv")
_TOK = re.compile(r"[a-z0-9]+")


def norm_url(u):
    try:
        p = urlsplit((u or "").strip())
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host + p.path.rstrip("/")
    except Exception:
        return (u or "").strip()


def load_detector(batch):
    lut = {}
    with open(DETECTOR, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if batch and row.get("batch") != batch:
                continue
            ct = row.get("Content type [gptzero]", "")
            if ct not in ("AI", "Mixed", "Human"):
                continue
            label = ct in ("AI", "Mixed")
            for key in (row.get("URL", ""), row.get("citation_url", "")):
                if key:
                    lut[norm_url(key)] = label
    return lut


def toks(s):
    return set(_TOK.findall((s or "").lower()))


def provenance(doc_id, url, lut):
    if str(doc_id).startswith("gen_"):
        return "answer_derived"
    lab = lut.get(norm_url(url))
    if lab is None:
        return "unlabeled"
    return "ai_original" if lab else "human_original"


def mean_ci(vals, n_boot=1000, seed=0):
    a = np.array(vals, float)
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)]
    return a.mean(), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main(path, batch="January_2026"):
    print(f"File: {path.split('/')[-1]}  | detector batch={batch}")
    lut = load_detector(batch)
    d = json.load(open(path))
    qs = d["questions"]
    print(f"{len(qs)} questions; {d['experiment_metadata'].get('num_iterations')} iterations")

    # collect per-doc records over all questions/rounds
    recs = []  # dict: question, round, prov, citation_rate, length, position, lexrel
    for q in qs:
        qtok = toks(q["question_text"])
        for it in q["iterations"]:
            rnd = it["iteration_number"]
            docs = it["documents"]
            # citation counts this round, aggregated over runs
            cc = {}
            tot = 0
            for run in it["runs"]:
                for c in (run.get("citations") or []):
                    cc[c["doc_id"]] = cc.get(c["doc_id"], 0) + 1
                    tot += 1
            if tot == 0:
                continue
            for pos, doc in enumerate(docs):
                prov = provenance(doc["doc_id"], doc.get("url"), lut)
                dt = toks(doc.get("text", ""))
                lexrel = len(qtok & dt) / len(qtok) if qtok else 0.0
                recs.append({
                    "question": q["question_id"], "round": rnd, "prov": prov,
                    "citation_rate": cc.get(doc["doc_id"], 0) / tot,
                    "length": len(doc.get("text", "")), "position": pos, "lexrel": lexrel,
                })

    # per-round availability + citation rate by group
    rounds = sorted({r["round"] for r in recs})
    print(f"\n{'rnd':>3} {'%AI_avail':>9} {'cr_self':>8} {'cr_aiorig':>9} {'cr_human':>9}")
    for rnd in rounds:
        rr = [r for r in recs if r["round"] == rnd]
        n = len(rr)
        avail_ai = sum(r["prov"] == "answer_derived" for r in rr) / n if n else 0
        def cr(g):
            v = [r["citation_rate"] for r in rr if r["prov"] == g]
            return np.mean(v) if v else float("nan")
        print(f"{rnd:>3} {avail_ai*100:>8.1f}% {cr('answer_derived'):>8.3f} "
              f"{cr('ai_original'):>9.3f} {cr('human_original'):>9.3f}")

    # mixed window = rounds where all three groups co-occur with originals present
    mixed = [r for r in recs if r["round"] >= 1]
    has_orig = {rr["round"] for rr in mixed if rr["prov"] in ("ai_original", "human_original")}
    mixed = [r for r in mixed if r["round"] in has_orig]
    print(f"\nMixed-window rounds (originals still present): {sorted(has_orig)}  | {len(mixed)} doc-rounds")
    print("=== Citation rate by provenance (pooled over mixed window) ===")
    for g in ("answer_derived", "ai_original", "human_original"):
        m, lo, hi = mean_ci([r["citation_rate"] for r in mixed if r["prov"] == g])
        n = sum(r["prov"] == g for r in mixed)
        print(f"  {g:<15} n={n:<6} mean={m:.4f}  CI [{lo:.4f}, {hi:.4f}]")

    # proxy OLS over mixed window (exclude unlabeled)
    sample = [r for r in mixed if r["prov"] != "unlabeled"]
    L = np.array([r["length"] for r in sample], float)
    Lz = (L - L.mean()) / (L.std() + 1e-9)
    X = np.column_stack([
        np.ones(len(sample)),
        [1.0 if r["prov"] == "answer_derived" else 0.0 for r in sample],
        [1.0 if r["prov"] == "ai_original" else 0.0 for r in sample],
        Lz,
        [r["position"] for r in sample],
        [r["lexrel"] for r in sample],
    ])
    y = np.array([r["citation_rate"] for r in sample], float)
    names = ["const", "is_answer_derived", "is_ai_original", "length_z", "position", "lexrel"]
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    rng = np.random.default_rng(0)
    boots = np.zeros((1000, X.shape[1]))
    for i in range(1000):
        idx = rng.integers(0, len(X), len(X))
        cb, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        boots[i] = cb
    print(f"\n=== Proxy OLS: citation_rate ~ provenance + length + position + lexrel (n={len(sample)}) ===")
    for j, nm in enumerate(names):
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        flag = ""
        if nm in ("is_answer_derived", "is_ai_original"):
            p = 2 * min((boots[:, j] > 0).mean(), (boots[:, j] < 0).mean())
            flag = f"  p={p:.4f}"
        print(f"  {nm:<18} {coefs[j]:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]{flag}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "January_2026")
