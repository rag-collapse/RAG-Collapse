"""Pure-Python tests (no GPU) for the rewrite distractor's Q1 fix in pipeline/misinfo.py:
gold-leak exclusion + substitution fallback. Run from the repo root:

    /c/Users/riddh/anaconda3/envs/rag-collapse/python.exe -m pytest base_hotpotqa_distractors/ -q
"""
import os
import re
import sys
import json
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import pipeline.misinfo as M


class FakeLLM:
    """Routes by judge/prompt type; the discovery judge branches on a marker in the
    (rewritten) document so we can drive the three rewrite outcomes deterministically."""
    def inference_batch(self, conversations):
        out = []
        for conv in conversations:
            sysc, usr = conv[0]["content"], conv[1]["content"]
            if "propose alternative" in sysc:                      # substitute proposal (fallback)
                out.append('["Lebanon", "Keene", "Concord"]')
            elif "subtly rewrites" in sysc:                        # rewrite: echo passage (keeps marker + gold)
                m = re.search(r"Original passage:\n(.*?)\n\nRewrite", usr, re.DOTALL)
                out.append(m.group(1) if m else usr)
            elif "fact-checking judge" in sysc:                    # discovery: branch on document marker
                if "GOODMARK" in usr:
                    out.append('{"asserted_answer": "Claremont", "contains_error": true}')
                elif "GOLDMARK" in usr:                            # judge re-extracts the gold (gold-leak)
                    out.append('{"asserted_answer": "Newport", "contains_error": true}')
                else:                                              # NOERRMARK: no usable wrong claim
                    out.append('{"asserted_answer": null, "contains_error": false}')
            else:
                out.append(usr)
        return out


def _make_state(qid, question, docs):
    return SimpleNamespace(query_id=qid, question_text=question,
                           current_docs=docs, question_obj={"iterations": []})


def _gt_file(tmp_path, mapping):
    p = tmp_path / "gt.json"
    p.write_text(json.dumps([{"_id": k, "answer": v} for k, v in mapping.items()]))
    return str(p)


def test_rewrite_gold_leak_excluded_and_substitution_fallback(tmp_path):
    """One doc rewrites to a real wrong entity (kept); one leaks the gold; one produces no
    error. The leaked + no-error docs must fall back to DISTINCT substituted wrong entities,
    so every corrupted doc ends up with a non-empty, non-gold, distinct entity."""
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    docs = [
        {"doc_id": "corpus_1", "text": "GOODMARK Newport is the county seat."},
        {"doc_id": "corpus_2", "text": "GOLDMARK Newport is the county seat."},
        {"doc_id": "corpus_3", "text": "NOERRMARK Newport is the county seat."},
    ]
    s = _make_state("q1", "What is the county seat?", docs)
    ctrl = M.DistractorController(mode="rewrite", fraction=1.0, gt_file=gt, doc_llm=FakeLLM(), seed=42)
    ctrl.prepare_and_apply([s])
    rec = ctrl.records["q1"]

    assert rec.eligible and rec.n_corrupted == 3
    ents = [d["injected_entity"] for d in rec.distractors]
    norms = [M.normalize(e) for e in ents]

    assert all(ents), f"a corrupted doc has no wrong entity: {ents}"            # fallback filled every doc
    assert "newport" not in norms, f"gold leaked into seeded set: {ents}"       # gold-leak excluded
    assert len(set(norms)) == len(norms), f"entities not distinct: {ents}"      # DIVERSE

    good = [d for d in rec.distractors if d["status"] == "ok"]
    fb = [d for d in rec.distractors if d["status"].startswith("fallback")]
    assert len(good) == 1 and good[0]["injected_entity"] == "Claremont"
    assert len(fb) == 2, f"expected 2 fallback docs, got {[d['status'] for d in rec.distractors]}"


def test_rewrite_fraction_zero_is_off():
    """Sanity: the controller rejects fraction 0 (the pipeline guard keeps it byte-identical;
    the controller is only built when fraction > 0)."""
    import pytest
    with pytest.raises(ValueError):
        M.DistractorController(mode="rewrite", fraction=0.0, gt_file="x", doc_llm=FakeLLM(), seed=1)
