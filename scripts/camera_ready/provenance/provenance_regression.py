"""
Self-reference vs. AI-content effect on citation rate (Qwen2.5-14B).

Quality-controlled test of whether the model over-cites its own generations beyond
what reference quality and AI authorship explain. Input is qwen_round1_scored.jsonl
(build_qwen_scoring_input.py -> score_references_qwen.py on the all_experiments
Qwen2.5-14B Replace-One round-1 references), plus the GPTZero detector labels.

Three provenance groups:
  answer_derived : the model's own prior answer (doc_id starts with gen_; is_ai==True)
  ai_original    : original web reference, detector says AI/Mixed
  human_original : original web reference, detector says Human

We report (a) citation rate by group, (b) key pairwise contrasts with a
question-clustered bootstrap, (c) an OLS of citation_rate on is_answer_derived
and is_ai_original controlling for the 8 quality dimensions, and (d) the
high-quality-only contrast (refs scoring 5/5 on direct_answer/organization/relevance).
"""

import csv
import json
import os
import sys
from urllib.parse import urlsplit

import numpy as np

DIMENSIONS = ["overall", "relevance", "accuracy", "thoroughness",
              "specificity", "up_to_date", "organization", "direct_answer"]

HERE = os.path.dirname(os.path.abspath(__file__))
SCORED = os.path.join(HERE, "qwen_round1_scored.jsonl")
DETECTOR = os.path.join(HERE, "ai_detector.csv")


def norm_url(u):
    try:
        p = urlsplit(u.strip())
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host + p.path.rstrip("/")
    except Exception:
        return u.strip()


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


def classify(rows, lut):
    stats = {"answer_derived": 0, "ai_original": 0, "human_original": 0, "unlabeled": 0}
    out = []
    for r in rows:
        rec = dict(r)
        if r["is_ai"]:
            rec["provenance"] = "answer_derived"
        else:
            lab = lut.get(norm_url(r["url"]))
            rec["provenance"] = "unlabeled" if lab is None else ("ai_original" if lab else "human_original")
        stats[rec["provenance"]] += 1
        out.append(rec)
    return out, stats


def mean_ci(vals, n_boot=2000, seed=0):
    a = np.array(vals, float)
    rng = np.random.default_rng(seed)
    boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)]
    return a.mean(), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def diff_ci_clustered(a, b, n_boot=2000, seed=0):
    """Difference in mean citation_rate (a-b), question-clustered bootstrap, two-sided p vs 0."""
    abq, bbq = {}, {}
    for r in a:
        abq.setdefault(r["question"], []).append(r["citation_rate"])
    for r in b:
        bbq.setdefault(r["question"], []).append(r["citation_rate"])
    qs = np.array(sorted(set(abq) | set(bbq)))
    point = np.mean([r["citation_rate"] for r in a]) - np.mean([r["citation_rate"] for r in b])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = qs[rng.integers(0, len(qs), len(qs))]
        av = [v for q in s for v in abq.get(q, [])]
        bv = [v for q in s for v in bbq.get(q, [])]
        if av and bv:
            boots.append(np.mean(av) - np.mean(bv))
    boots = np.array(boots)
    p = 2 * min((boots > 0).mean(), (boots < 0).mean())
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(p)


def regression(ann, n_boot=2000):
    sample = [r for r in ann if r["provenance"] != "unlabeled"]
    rows = []
    for r in sample:
        rows.append([1.0,
                     1.0 if r["provenance"] == "answer_derived" else 0.0,
                     1.0 if r["provenance"] == "ai_original" else 0.0]
                    + [float(r[d]) for d in DIMENSIONS])
    X = np.array(rows)
    y = np.array([r["citation_rate"] for r in sample], float)
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    names = ["const", "is_answer_derived", "is_ai_original"] + DIMENSIONS
    rng = np.random.default_rng(0)
    boots = np.zeros((n_boot, X.shape[1]))
    for i in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        cb, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        boots[i] = cb
    print(f"\n=== OLS: citation_rate ~ const + is_answer_derived + is_ai_original + 8 dims (n={len(sample)}) ===")
    for j, nm in enumerate(names):
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        flag = ""
        if nm in ("is_answer_derived", "is_ai_original"):
            p = 2 * min((boots[:, j] > 0).mean(), (boots[:, j] < 0).mean())
            flag = f"  p={p:.4f}"
        print(f"  {nm:<18} {coefs[j]:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]{flag}")


def high_quality_contrast(ann):
    hq = [r for r in ann if r["provenance"] != "unlabeled"
          and r["direct_answer"] == 5 and r["organization"] == 5 and r["relevance"] == 5]
    ad = [r["citation_rate"] for r in hq if r["provenance"] == "answer_derived"]
    orig = [r["citation_rate"] for r in hq if r["provenance"] in ("ai_original", "human_original")]
    print(f"\n=== High-quality-only (5/5 direct_answer & organization & relevance), n={len(hq)} ===")
    if ad and orig:
        ma, mo = np.mean(ad), np.mean(orig)
        print(f"  answer_derived: {ma:.3f} (n={len(ad)})   original: {mo:.3f} (n={len(orig)})   "
              f"gap={ma/mo:.1f}x" if mo > 0 else "")


def main(scored=SCORED, batch="January_2026"):
    rows = [json.loads(l) for l in open(scored)]
    rnds = sorted({r.get("round") for r in rows})
    print(f"File: {scored.split('/')[-1]}")
    print(f"Loaded {len(rows)} scored references; {len({r['question'] for r in rows})} questions; rounds={rnds}")
    lut = load_detector(batch)
    print(f"Detector batch={batch}: {len(lut)} URL labels")
    ann, stats = classify(rows, lut)
    print("Provenance counts:", stats)
    labeled = stats["ai_original"] + stats["human_original"]
    n_orig = labeled + stats["unlabeled"]
    if n_orig:
        print(f"Originals labeled by detector: {labeled}/{n_orig} ({labeled/n_orig:.0%}); "
              f"AI share among labeled: {stats['ai_original']/labeled:.0%}")

    print("\n=== Mean quality by provenance ===")
    for g in ("answer_derived", "ai_original", "human_original"):
        recs = [r for r in ann if r["provenance"] == g]
        if recs:
            parts = " ".join(f"{d[:4]}={np.mean([r[d] for r in recs]):.2f}" for d in DIMENSIONS)
            print(f"  {g:<15} n={len(recs):<4} {parts}")

    print("\n=== Citation rate by provenance ===")
    for g in ("answer_derived", "ai_original", "human_original"):
        rates = [r["citation_rate"] for r in ann if r["provenance"] == g]
        if rates:
            m, lo, hi = mean_ci(rates)
            print(f"  {g:<15} n={len(rates):<4} mean={m:.4f}  CI [{lo:.4f}, {hi:.4f}]")

    recs = {g: [r for r in ann if r["provenance"] == g] for g in
            ("answer_derived", "ai_original", "human_original")}
    print("\n=== Key contrasts (question-clustered bootstrap) ===")
    d, lo, hi, p = diff_ci_clustered(recs["ai_original"], recs["human_original"])
    print(f"  ai_original - human_original  : {d:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  p={p:.4f}  <- AI-ness alone")
    d, lo, hi, p = diff_ci_clustered(recs["answer_derived"], recs["human_original"])
    print(f"  answer_derived - human_orig.  : {d:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  p={p:.4f}")
    d, lo, hi, p = diff_ci_clustered(recs["answer_derived"], recs["ai_original"])
    print(f"  answer_derived - ai_original  : {d:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  p={p:.4f}  <- self beyond AI-ness")

    high_quality_contrast(ann)
    regression(ann)


if __name__ == "__main__":
    scored = sys.argv[1] if len(sys.argv) > 1 else SCORED
    batch = sys.argv[2] if len(sys.argv) > 2 else "January_2026"
    main(scored, batch)
