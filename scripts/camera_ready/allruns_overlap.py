"""Overlap-family citation analyses (R1 placebo, R2 over-citation ratio, prefilter amplification)
across every run, computed from answers + documents only. No citations needed, no generation, so
this runs on models and variants that never logged citations (Llama, DeepSeek, agentic, ...).

Per run it reports, at round 1 and pooled:
  - R2 over-citation ratio = selfgen-share(cited) / selfgen-share(context), cited = overlap >= tau
  - prefilter amplification = selfgen-share(top-2 overlap eligible) / selfgen-share(context)
  - R1 placebo FP at tau: overlap-rule hit rate on docs NEVER in this answer's context,
    split into within-question descendants (self-gen from a later round) vs cross-question.

Agentic runs list a doc multiple times per round (repeat tool retrievals); docs are de-duplicated
per round by doc_id. Runs whose answers are too short (HotpotQA) are flagged, not trusted.
"""
import json, re, sys, os, random
random.seed(3)
WORD = re.compile(r"[A-Za-z0-9']+")
TAU = 0.20
TOP_M = 2

def tok(t):
    return {w.lower() for w in WORD.findall(t or "") if len(w) >= 4}

def ov(a, dset):
    return len(a & dset) / max(1, len(a)) if a else 0.0

def is_ai(doc_id, url, it):
    doc_id = str(doc_id or "").lower(); url = str(url or "").lower()
    return doc_id.startswith("gen_") or url == "model_generated" or (isinstance(it, int) and it > 0 and not doc_id.startswith("ref_"))

def analyze(path):
    d = json.load(open(path))
    md = d.get("experiment_metadata", {})
    # gather, per question: round -> deduped docs [(doc_id, toks, is_ai)], answers[list], and
    # a global pool of (question, doc) for the cross-question placebo.
    Q = []
    all_docs = []  # (qi, doc_id, toks, is_ai)
    ans_len = []
    for qi, q in enumerate(d["questions"]):
        its = {}
        for it in q["iterations"]:
            r = it["iteration_number"]
            seen = {}
            for x in (it.get("documents") or []):
                did = x.get("doc_id")
                if did and did not in seen:
                    seen[did] = (did, tok(x.get("text")), is_ai(did, x.get("url"), x.get("iteration")))
            docs = list(seen.values())
            answers = [run.get("answer", "") for run in (it.get("runs") or [])]
            its[r] = {"docs": docs, "answers": answers}
            for did, dt, ai in docs:
                all_docs.append((qi, did, dt, ai))
            ans_len += [len(tok(a)) for a in answers]
        Q.append(its)

    def octa(rounds):
        """over-citation ratio + prefilter amplification over the given rounds."""
        ctx_sg = ctx_n = cit_sg = cit_n = top_sg = top_n = 0
        for qi, its in enumerate(Q):
            for r in rounds:
                if r not in its:
                    continue
                docs = its[r]["docs"]
                for did, dt, ai in docs:
                    ctx_n += 1; ctx_sg += int(ai)
                for a in its[r]["answers"]:
                    at = tok(a)
                    scored = sorted(docs, key=lambda x: ov(at, x[1]), reverse=True)
                    for did, dt, ai in scored[:TOP_M]:
                        top_n += 1; top_sg += int(ai)
                    for did, dt, ai in docs:
                        if ov(at, dt) >= TAU:
                            cit_n += 1; cit_sg += int(ai)
        cs = ctx_sg / ctx_n if ctx_n else float("nan")
        return dict(ctx=cs,
                    ratio=(cit_sg / cit_n) / cs if cit_n and cs else float("nan"),
                    amp=(top_sg / top_n) / cs if top_n and cs else float("nan"))

    # R1 placebo at round 1: overlap-rule hits on docs NEVER in this answer's round-1 context
    desc_hit = desc_n = xq_sg_hit = xq_sg_n = xq_h_hit = xq_h_n = 0
    for qi, its in enumerate(Q):
        if 1 not in its:
            continue
        ctx_ids = {did for did, _, _ in its[1]["docs"]}
        later_self = [(did, dt) for r2 in its if r2 > 1 for did, dt, ai in its[r2]["docs"] if ai and did not in ctx_ids]
        for a in its[1]["answers"]:
            at = tok(a)
            for did, dt in later_self:  # within-Q descendant self-gen never in this context
                desc_n += 1; desc_hit += int(ov(at, dt) >= TAU)
            for _ in range(3):  # cross-question sample
                oj, od, odt, oai = random.choice(all_docs)
                if oj == qi:
                    continue
                if oai:
                    xq_sg_n += 1; xq_sg_hit += int(ov(at, odt) >= TAU)
                else:
                    xq_h_n += 1; xq_h_hit += int(ov(at, odt) >= TAU)
    r1 = dict(desc=desc_hit / desc_n if desc_n else float("nan"),
              xq_self=xq_sg_hit / xq_sg_n if xq_sg_n else float("nan"),
              xq_human=xq_h_hit / xq_h_n if xq_h_n else float("nan"))

    all_rounds = sorted({r for its in Q for r in its if r > 0})
    return dict(variant=md.get("pipeline_variant"), model=md.get("model"),
                citations=md.get("citations_enabled"),
                med_answer_tok=sorted(ans_len)[len(ans_len)//2] if ans_len else 0,
                r1_pooled=octa(all_rounds), r1_round1=octa([1]), placebo=r1)

if __name__ == "__main__":
    print(f"{'model':16s} {'variant':12s} {'medAns':>6} {'ctx1%':>6} {'ratio1':>7} {'amp1':>6} {'ratioP':>7} | placebo desc/xqSelf/xqHuman")
    for p in sys.argv[1:]:
        try:
            r = analyze(p)
        except Exception as e:
            print(f"ERR {p}: {e}"); continue
        a = r["r1_round1"]; ap = r["r1_pooled"]; pb = r["placebo"]
        print(f"{str(r['model']):16s} {str(r['variant']):12s} {r['med_answer_tok']:6d} "
              f"{100*a['ctx']:6.1f} {a['ratio']:7.2f} {a['amp']:6.2f} {ap['ratio']:7.2f} | "
              f"{100*pb['desc']:.0f}% / {100*pb['xq_self']:.0f}% / {100*pb['xq_human']:.0f}%")
