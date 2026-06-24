"""Pure-Python tests for the misinformation-injection machinery (no GPU / no servers).

Run from the repo root:  python -m pytest tests/test_misinfo.py -q
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline import misinfo as M  # noqa: E402


# ── text helpers ─────────────────────────────────────────────────────────────
def test_normalize_matches_hotpot_eval():
    """Guard against drift between misinfo.normalize and the eval normalizer."""
    from hotpot_evaluation import _normalize_answer
    for s in ["The Newport, NH!", "  yes  ", "An Apple", "Bose QC-45", "", "U.S.A."]:
        assert M.normalize(s) == _normalize_answer(s)


def test_contains_entity_case_and_punct():
    assert M.contains_entity("I recommend the Bose QC45 headphones.", "bose qc45")
    assert M.contains_entity("The county seat is Newport.", "Newport")
    assert not M.contains_entity("Bose headphones are great", "Sony")
    assert not M.contains_entity("anything", "")


def test_substitute_in_text_boundaries_and_count():
    new, n = M.substitute_in_text("The seat is Newport, near Newport.", "Newport", "Claremont")
    assert n == 2 and "Claremont" in new and "Newport" not in new
    # boundary: do not replace inside a larger word
    new2, n2 = M.substitute_in_text("Newportland is unrelated", "Newport", "Claremont")
    assert n2 == 0 and new2 == "Newportland is unrelated"
    # case-insensitive match, substitute casing preserved
    new3, n3 = M.substitute_in_text("answer: newport", "Newport", "Claremont")
    assert n3 == 1 and "Claremont" in new3


def test_validate_and_choose_substitute():
    assert M.validate_substitute("Claremont", "Newport", "What is the county seat?")
    assert not M.validate_substitute("Newport", "Newport", "q")          # equal
    assert not M.validate_substitute("Newport News", "Newport", "q")     # substring overlap
    assert not M.validate_substitute("Paris", "Newport", "Best things in Paris?")  # in question
    chosen = M.choose_substitute(["Newport", "Newport News", "Claremont"], "Newport", "q")
    assert chosen == "Claremont"
    assert M.choose_substitute(["Newport"], "Newport", "q") is None


def test_realized_status():
    assert M.realized_status("seat is Claremont", "Newport", "Claremont") == "ok"
    assert M.realized_status("Claremont, formerly Newport", "Newport", "Claremont") == "leaky_gold"
    assert M.realized_status("a faithful article", "Newport", "Claremont") == "not_realized"


def test_parse_json_variants():
    assert M.parse_json('```json\n["a", "b"]\n```') == ["a", "b"]
    assert M.parse_json('["x"]') == ["x"]
    assert M.parse_json('noise {"k": 1} trailing') == {"k": 1}
    assert M.parse_json("not json at all") is None
    assert M.parse_json(None) is None


def test_load_ground_truth_both_formats(tmp_path):
    p_list = tmp_path / "native.json"
    p_list.write_text(json.dumps([{"_id": "a", "answer": "X"}, {"_id": "b", "answer": "Y"}]))
    assert M.load_ground_truth(str(p_list)) == {"a": "X", "b": "Y"}
    p_map = tmp_path / "flat.json"
    p_map.write_text(json.dumps({"a": "X"}))
    assert M.load_ground_truth(str(p_map)) == {"a": "X"}


def test_bridge_entity_for():
    rec = {
        "type": "bridge",
        "answer": "Newport",
        "supporting_facts": [["Sullivan County", 0], ["East Lempster", 1]],
        "context": [
            ["Sullivan County", ["Its county seat is Newport."]],
            ["East Lempster", ["East Lempster is a village in New Hampshire."]],
        ],
    }
    # bridge = supporting title WITHOUT the gold answer span
    assert M.bridge_entity_for(rec, "Newport") == "East Lempster"
    assert M.bridge_entity_for({"type": "comparison"}, "x") is None


def test_injection_record_roundtrip():
    rec = M.InjectionRecord(
        mode="counterfactual", target="final_answer", inject_round=1,
        inject_every_round=False, seed=42, gold_answer="Newport", eligible=True,
        injected_entity="Claremont",
    )
    d = rec.to_dict()
    assert d["mode"] == "counterfactual" and d["injected_entity"] == "Claremont"
    assert d["status"] == "pending" and d["injected_doc_ids"] == []


# ── controller end-to-end with a fake LLM ────────────────────────────────────
class FakeLLM:
    """Returns canned outputs keyed by which judge/synthesis prompt it sees."""
    def inference_batch(self, conversations):
        out = []
        for conv in conversations:
            sysc, usr = conv[0]["content"], conv[1]["content"]
            if "propose alternative" in sysc:
                out.append('["Claremont", "Lebanon", "Keene"]')
            elif "fact-checking judge" in sysc:
                out.append('{"asserted_answer": "Claremont", "false_claims": ["seat is Claremont"], "contains_error": true}')
            elif "multi-hop reasoning" in sysc:
                out.append('{"implied_answer": "North Haverhill"}')
            elif "Wikipedia-style" in sysc:  # diverse_synth distractor doc: echo the wrong answer
                m = re.search(r"correct answer to the question below is (.+?)\.", usr)
                out.append(f"A Wikipedia-style lead paragraph. The answer is {m.group(1).strip()}." if m else usr)
            else:  # create-document: echo the (possibly substituted) answer back
                m = re.search(r"Here is the answer:\n(.*?)\n\nInstructions", usr, re.DOTALL)
                out.append(m.group(1) if m else usr)
        return out


def _make_state(qid, question):
    return SimpleNamespace(query_id=qid, question_text=question, question_obj={"iterations": []})


def _gt_file(tmp_path, mapping):
    p = tmp_path / "gt.json"
    p.write_text(json.dumps([{"_id": k, "answer": v} for k, v in mapping.items()]))
    return str(p)


def test_controller_counterfactual_final_answer(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    states = [_make_state("q1", "What is the county seat of the county containing East Lempster?")]
    ctrl = M.MisinfoController(
        mode="counterfactual", target_mode="final_answer", inject_round=0,
        inject_every_round=False, gt_file=gt, doc_llm=FakeLLM(), seed=42,
    )
    ctrl.prepare(states)
    rec = ctrl.records["q1"]
    assert rec.eligible and rec.injected_entity == "Claremont"

    # inject round: gold-bearing answer gets substituted before synthesis
    conv = ctrl.make_doc_conversation(states[0], "The county seat is Newport.", it=0)
    assert "Claremont" in conv[1]["content"] and "Newport" not in conv[1]["content"]
    doc_text = FakeLLM().inference_batch([conv])[0]
    ctrl.record_injection([(states[0], [doc_text], ["gen_1_0"])], it=0)
    assert rec.status == "ok" and rec.injected_doc_ids == ["gen_1_0"]

    # non-inject round: faithful, answer untouched
    conv2 = ctrl.make_doc_conversation(states[0], "The county seat is Newport.", it=5)
    assert "Newport" in conv2[1]["content"] and "Claremont" not in conv2[1]["content"]


def test_controller_yes_no_skipped(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "yes"})
    states = [_make_state("q1", "Are both directors American?")]
    ctrl = M.MisinfoController(
        mode="counterfactual", target_mode="final_answer", inject_round=0,
        inject_every_round=False, gt_file=gt, doc_llm=FakeLLM(), seed=1,
    )
    ctrl.prepare(states)
    rec = ctrl.records["q1"]
    assert not rec.eligible and rec.status == "skipped_yes_no"
    # ineligible → faithful conversation regardless of round
    conv = ctrl.make_doc_conversation(states[0], "yes", it=0)
    assert conv == M.get_create_document_conversation(question=states[0].question_text, answer="yes")


def test_controller_gold_absent_in_answer(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    states = [_make_state("q1", "County seat question?")]
    ctrl = M.MisinfoController(
        mode="counterfactual", target_mode="final_answer", inject_round=0,
        inject_every_round=False, gt_file=gt, doc_llm=FakeLLM(), seed=7,
    )
    ctrl.prepare(states)
    conv = ctrl.make_doc_conversation(states[0], "I am not sure about this one.", it=0)
    # no gold span to substitute → falls back to faithful
    assert conv == M.get_create_document_conversation(
        question=states[0].question_text, answer="I am not sure about this one.")
    ctrl.record_injection([(states[0], ["a faithful doc"], ["gen_1_0"])], it=0)
    assert ctrl.records["q1"].status == "gold_absent_in_answer"


def test_controller_freeform_discovery(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    states = [_make_state("q1", "County seat question?")]
    ctrl = M.MisinfoController(
        mode="freeform", target_mode="final_answer", inject_round=0,
        inject_every_round=False, gt_file=gt, doc_llm=FakeLLM(), seed=0,
    )
    ctrl.prepare(states)
    assert ctrl.records["q1"].eligible
    conv = ctrl.make_doc_conversation(states[0], "The seat is Newport.", it=0)
    assert "factually INCORRECT claim about the answer" in conv[1]["content"]
    ctrl.record_injection([(states[0], ["some doc claiming Claremont"], ["gen_1_0"])], it=0)
    rec = ctrl.records["q1"]
    assert rec.status == "ok" and rec.discovered_claims["contains_error"] is True


def test_controller_hop_requires_native(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    with pytest.raises(ValueError):
        M.MisinfoController(
            mode="counterfactual", target_mode="intermediate_hop", inject_round=0,
            inject_every_round=False, gt_file=gt, doc_llm=FakeLLM(), seed=0,
            native_file=None,
        )


# ── initial-doc distractor helpers ────────────────────────────────────────────
def test_n_to_corrupt():
    assert M.n_to_corrupt(0.5, 10) == 5
    assert M.n_to_corrupt(0.0, 10) == 0
    assert M.n_to_corrupt(0.5, 0) == 0
    assert M.n_to_corrupt(1.0, 4) == 4
    assert M.n_to_corrupt(0.3, 10) == 3
    assert M.n_to_corrupt(2.0, 4) == 4          # clamped to k


def test_select_distractor_indices_prefers_gold_bearing():
    import random as _r
    docs = [
        {"doc_id": "c1", "text": "The seat is Newport."},
        {"doc_id": "c2", "text": "Unrelated geography text."},
        {"doc_id": "c3", "text": "Newport history and trivia."},
        {"doc_id": "c4", "text": "Nothing relevant here."},
    ]
    idxs = M.select_distractor_indices(docs, "Newport", 2, _r.Random(0))
    # both chosen docs must be the gold-bearing ones (indices 0 and 2)
    assert set(idxs) == {0, 2}
    assert M.select_distractor_indices(docs, "Newport", 0, _r.Random(0)) == []


def test_choose_distinct_substitutes():
    cands = ["Newport", "Claremont", "Claremont", "Lebanon", "Newport News"]
    out = M.choose_distinct_substitutes(cands, "Newport", "q", 2)
    assert out == ["Claremont", "Lebanon"]      # skips equal/dup/substring-overlap, distinct
    assert M.choose_distinct_substitutes(["Newport"], "Newport", "q", 3) == []


def test_distractor_record_roundtrip():
    rec = M.DistractorRecord(fraction=0.5, mode="substitution", gold_answer="Newport", eligible=True)
    d = rec.to_dict()
    assert d["mode"] == "substitution" and d["eligible"] is True
    assert d["distractors"] == [] and d["status"] == "pending"


def test_distractor_controller_substitution_diverse(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    s = _make_state("q1", "What is the county seat?")
    s.current_docs = [
        {"doc_id": "corpus_1", "text": "The seat is Newport."},
        {"doc_id": "corpus_2", "text": "Newport is the county seat of the area."},
        {"doc_id": "corpus_3", "text": "Unrelated text about mountains."},
        {"doc_id": "corpus_4", "text": "More about Newport and its history."},
    ]
    ctrl = M.DistractorController(
        mode="substitution", fraction=0.5, gt_file=gt, doc_llm=FakeLLM(), seed=42)
    ctrl.prepare_and_apply([s])
    rec = ctrl.records["q1"]
    assert rec.eligible and rec.status == "ok"
    assert rec.n_corrupted == 2                          # round(0.5*4)
    ents = [d["injected_entity"] for d in rec.distractors]
    assert len(ents) == 2 and len(set(ents)) == 2        # DIVERSE: distinct wrong entities
    # corrupted docs were mutated in place + flagged; gold replaced by the wrong entity
    corrupted = [d for d in s.current_docs if d.get("distractor")]
    assert len(corrupted) == 2
    for d in corrupted:
        assert not M.contains_entity(d["text"], "Newport")
        assert any(M.contains_entity(d["text"], e) for e in ents)
    # only gold-bearing docs (1,2,4) were chosen, never the unrelated corpus_3
    assert all(d["doc_id"] in {"corpus_1", "corpus_2", "corpus_4"} for d in rec.distractors)


def test_distractor_controller_off_by_default_yesno(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "yes"})
    s = _make_state("q1", "Are both directors American?")
    s.current_docs = [{"doc_id": "corpus_1", "text": "Some doc."}]
    ctrl = M.DistractorController(
        mode="substitution", fraction=0.5, gt_file=gt, doc_llm=FakeLLM(), seed=1)
    ctrl.prepare_and_apply([s])
    rec = ctrl.records["q1"]
    assert not rec.eligible and rec.status == "skipped_yes_no"
    assert not any(d.get("distractor") for d in s.current_docs)   # untouched


def test_distractor_controller_native_noise(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    native_p = tmp_path / "native.json"
    native_p.write_text(json.dumps([{
        "_id": "q1", "answer": "Newport", "type": "bridge",
        "supporting_facts": [["Sullivan County", 0]],
        "context": [
            ["Sullivan County", ["Its county seat is Newport."]],
            ["Decoy A", ["Some unrelated paragraph about rivers."]],
            ["Decoy B", ["Another unrelated paragraph about mountains."]],
        ],
    }]))
    s = _make_state("q1", "What is the county seat?")
    s.current_docs = [
        {"doc_id": "corpus_1", "text": "The seat is Newport."},
        {"doc_id": "corpus_2", "text": "Newport history."},
    ]
    ctrl = M.DistractorController(
        mode="native_noise", fraction=0.5, gt_file=gt, doc_llm=FakeLLM(), seed=3,
        native_file=str(native_p))
    ctrl.prepare_and_apply([s])
    rec = ctrl.records["q1"]
    assert rec.eligible and rec.n_corrupted == 1
    corrupted = [d for d in s.current_docs if d.get("distractor")]
    assert len(corrupted) == 1
    # swapped in a real non-supporting paragraph (no asserted wrong entity)
    assert "unrelated paragraph" in corrupted[0]["text"]
    assert rec.distractors[0]["injected_entity"] == ""


def test_distractor_native_noise_requires_native_file(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    with pytest.raises(ValueError):
        M.DistractorController(
            mode="native_noise", fraction=0.5, gt_file=gt, doc_llm=FakeLLM(), seed=0,
            native_file=None)


# ── diverse_synth: coordinated distinct distractors, one synthesized doc each ──
class _GoldLeakLLM:
    """Proposal returns distinct candidates; create-document leaks the gold answer."""
    def inference_batch(self, conversations):
        out = []
        for conv in conversations:
            sysc = conv[0]["content"]
            if "propose alternative" in sysc:
                out.append('["Claremont", "Lebanon", "Keene"]')
            else:  # create-document → returns a doc still mentioning the gold (Newport)
                out.append("The county seat is Newport.")
        return out


class _GoldOnlyProposalLLM:
    """Proposal returns only the gold (no valid distinct substitute exists)."""
    def inference_batch(self, conversations):
        return ['["Newport"]' if "propose alternative" in c[0]["content"] else "doc"
                for c in conversations]


class _RefusalLLM:
    """Proposal returns distinct candidates; create-document REFUSES (asserts neither
    the wrong answer nor the gold) — simulates a strong model declining the falsehood."""
    def inference_batch(self, conversations):
        out = []
        for conv in conversations:
            if "propose alternative" in conv[0]["content"]:
                out.append('["Claremont", "Lebanon", "Keene"]')
            else:
                out.append("I'm sorry, but I can't help create that content.")
        return out


def test_distractor_controller_diverse_synth(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    s = _make_state("q1", "What is the county seat?")
    s.current_docs = [
        {"doc_id": "corpus_1", "text": "The seat is Newport."},
        {"doc_id": "corpus_2", "text": "Newport is the county seat of the area."},
        {"doc_id": "corpus_3", "text": "Unrelated text about mountains."},
        {"doc_id": "corpus_4", "text": "More about Newport and its history."},
    ]
    ctrl = M.DistractorController(
        mode="diverse_synth", fraction=0.5, gt_file=gt, doc_llm=FakeLLM(), seed=42)
    ctrl.prepare_and_apply([s])
    rec = ctrl.records["q1"]
    assert rec.eligible and rec.status == "ok"
    assert rec.n_corrupted == 2                          # round(0.5*4); 3 gold docs, none all-consumed
    ents = [d["injected_entity"] for d in rec.distractors]
    assert len(ents) == 2 and len(set(ents)) == 2        # DIVERSE: distinct wrong answers
    corrupted = [d for d in s.current_docs if d.get("distractor")]
    assert len(corrupted) == 2
    for d in corrupted:
        assert not M.contains_entity(d["text"], "Newport")    # gold replaced by the wrong answer
        assert any(M.contains_entity(d["text"], e) for e in ents)
    # gold preserved: at least one gold-bearing doc remains uncorrupted
    assert any((not d.get("distractor")) and M.contains_entity(d["text"], "Newport")
               for d in s.current_docs)


def test_diverse_synth_preserves_gold(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    s = _make_state("q1", "What is the county seat?")
    s.current_docs = [
        {"doc_id": "corpus_1", "text": "The seat is Newport."},
        {"doc_id": "corpus_2", "text": "Newport history and trivia."},
        {"doc_id": "corpus_3", "text": "Unrelated text about rivers."},
        {"doc_id": "corpus_4", "text": "Unrelated text about mountains."},
    ]
    # fraction 1.0 would corrupt ALL 4 docs (both gold docs included) → preservation kicks in
    ctrl = M.DistractorController(
        mode="diverse_synth", fraction=1.0, gt_file=gt, doc_llm=FakeLLM(), seed=42)
    ctrl.prepare_and_apply([s])
    rec = ctrl.records["q1"]
    assert rec.eligible and rec.n_corrupted == 3          # one gold doc dropped from the 4
    assert any("preserved gold doc" in n for n in rec.notes)
    kept = [d for d in s.current_docs if not d.get("distractor")]
    assert len(kept) == 1 and M.contains_entity(kept[0]["text"], "Newport")


def test_diverse_synth_yes_no_skipped(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "yes"})
    s = _make_state("q1", "Are both directors American?")
    s.current_docs = [{"doc_id": "corpus_1", "text": "Some doc."}]
    ctrl = M.DistractorController(
        mode="diverse_synth", fraction=0.5, gt_file=gt, doc_llm=FakeLLM(), seed=1)
    ctrl.prepare_and_apply([s])
    rec = ctrl.records["q1"]
    assert not rec.eligible and rec.status == "skipped_yes_no"
    assert not any(d.get("distractor") for d in s.current_docs)


def test_diverse_synth_no_valid_substitute(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    s = _make_state("q1", "What is the county seat?")
    s.current_docs = [
        {"doc_id": "corpus_1", "text": "The seat is Newport."},
        {"doc_id": "corpus_2", "text": "Newport history."},
        {"doc_id": "corpus_3", "text": "Unrelated."},
    ]
    ctrl = M.DistractorController(
        mode="diverse_synth", fraction=1.0, gt_file=gt, doc_llm=_GoldOnlyProposalLLM(), seed=5)
    ctrl.prepare_and_apply([s])
    rec = ctrl.records["q1"]
    assert rec.status == "no_valid_substitute" and not rec.eligible
    assert not any(d.get("distractor") for d in s.current_docs)   # docs untouched


def test_diverse_synth_gold_leak_replaced(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    s = _make_state("q1", "What is the county seat?")
    s.current_docs = [
        {"doc_id": "corpus_1", "text": "The seat is Newport."},
        {"doc_id": "corpus_2", "text": "Newport history."},
        {"doc_id": "corpus_3", "text": "Unrelated mountains."},
    ]
    # create-document leaks the gold → the doc is rejected and replaced by a clean templated
    # assertion of the wrong answer (no refusal text, no gold mention).
    ctrl = M.DistractorController(
        mode="diverse_synth", fraction=0.5, gt_file=gt, doc_llm=_GoldLeakLLM(), seed=42)
    ctrl.prepare_and_apply([s])
    rec = ctrl.records["q1"]
    assert rec.eligible and rec.n_corrupted >= 1
    corrupted = [d for d in s.current_docs if d.get("distractor")]
    assert corrupted
    for d, drec in zip(corrupted, rec.distractors):
        assert not M.contains_entity(d["text"], "Newport")            # gold leak removed
        assert M.contains_entity(d["text"], drec["injected_entity"])  # asserts the wrong answer
        assert drec["status"] == "fallback_template"


def test_diverse_synth_refusal_replaced(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    s = _make_state("q1", "What is the county seat?")
    s.current_docs = [
        {"doc_id": "corpus_1", "text": "The seat is Newport."},
        {"doc_id": "corpus_2", "text": "Newport history."},
        {"doc_id": "corpus_3", "text": "Unrelated mountains."},
    ]
    # doc-LLM refuses → not realized → replaced by a clean templated assertion of the wrong answer
    ctrl = M.DistractorController(
        mode="diverse_synth", fraction=0.5, gt_file=gt, doc_llm=_RefusalLLM(), seed=42)
    ctrl.prepare_and_apply([s])
    rec = ctrl.records["q1"]
    assert rec.eligible and rec.status == "ok" and rec.n_corrupted >= 1
    for d, drec in zip([x for x in s.current_docs if x.get("distractor")], rec.distractors):
        assert "sorry" not in d["text"].lower()                       # refusal text dropped
        assert M.contains_entity(d["text"], drec["injected_entity"])  # asserts the wrong answer
        assert not M.contains_entity(d["text"], "Newport")
        assert drec["status"] == "fallback_template"


def test_per_run_doc_subsets():
    docs = [
        {"doc_id": "g1", "text": "gold doc"},                       # clean
        {"doc_id": "c2", "text": "clean doc"},                      # clean
        {"doc_id": "d1", "text": "Claremont", "distractor": True},
        {"doc_id": "d2", "text": "Lebanon", "distractor": True},
        {"doc_id": "d3", "text": "Keene", "distractor": True},
    ]
    subs = M.per_run_doc_subsets(docs, num_runs=5)
    assert len(subs) == 5
    # each run = the 2 clean docs + exactly ONE distractor, rotating d1,d2,d3,d1,d2
    rotation = []
    for sub in subs:
        clean_ids = [d["doc_id"] for d in sub if not d.get("distractor")]
        dist_ids = [d["doc_id"] for d in sub if d.get("distractor")]
        assert clean_ids == ["g1", "c2"]
        assert len(dist_ids) == 1
        rotation.append(dist_ids[0])
    assert rotation == ["d1", "d2", "d3", "d1", "d2"]
    # 3 distinct distractors used across the first 3 runs -> diverse contexts
    assert len(set(rotation)) == 3


def test_per_run_doc_subsets_no_distractors():
    docs = [{"doc_id": "g1", "text": "x"}, {"doc_id": "c2", "text": "y"}]
    subs = M.per_run_doc_subsets(docs, num_runs=3)
    assert len(subs) == 3 and all([d["doc_id"] for d in s] == ["g1", "c2"] for s in subs)


def test_select_distractor_indices_avoid_gold():
    import random as _r
    docs = [
        {"doc_id": "g1", "text": "The seat is Newport.", "gold": True},  # tagged gold
        {"doc_id": "c2", "text": "Also mentions Newport."},             # contains gold answer
        {"doc_id": "d3", "text": "Unrelated rivers."},                  # non-gold
        {"doc_id": "d4", "text": "Unrelated mountains."},               # non-gold
    ]
    idxs = M.select_distractor_indices(docs, "Newport", 4, _r.Random(0), avoid_gold=True)
    assert set(idxs) == {2, 3}   # gold-tagged AND gold-containing docs are never selected


def test_distractor_controller_diverse_synth_avoid_gold(tmp_path):
    gt = _gt_file(tmp_path, {"q1": "Newport"})
    s = _make_state("q1", "What is the county seat?")
    s.current_docs = [
        {"doc_id": "g1", "text": "Newport is the seat.", "gold": True},
        {"doc_id": "g2", "text": "More on Newport.", "gold": True},
        {"doc_id": "d3", "text": "A distractor paragraph about rivers.", "distractor": True},
        {"doc_id": "d4", "text": "A distractor paragraph about hills.", "distractor": True},
    ]
    ctrl = M.DistractorController(
        mode="diverse_synth", fraction=1.0, gt_file=gt, doc_llm=FakeLLM(), seed=42, avoid_gold=True)
    ctrl.prepare_and_apply([s])
    rec = ctrl.records["q1"]
    assert rec.eligible and rec.status == "ok"
    # only the 2 non-gold docs were corrupted; the 2 gold paragraphs are untouched
    assert {d["doc_id"] for d in rec.distractors} == {"d3", "d4"}
    assert s.current_docs[0]["text"] == "Newport is the seat."
    assert s.current_docs[1]["text"] == "More on Newport."
    ents = [d["injected_entity"] for d in rec.distractors]
    assert len(set(ents)) == 2   # still distinct


def test_distractor_eval_metrics():
    from hotpot_evaluation import _distractor_metrics_for_iteration
    dist = {
        "eligible": True, "gold_answer": "Newport",
        "distractors": [
            {"doc_id": "corpus_1", "injected_entity": "Claremont"},
            {"doc_id": "corpus_2", "injected_entity": "Lebanon"},
        ],
    }
    iteration = {
        "documents": [
            {"doc_id": "corpus_1", "text": "...", "distractor": True},
            {"doc_id": "corpus_9", "text": "x"},
        ],
        "runs": [],
    }
    preds = ["It is Claremont", "Lebanon", "The seat is Newport", "Keene"]
    m = _distractor_metrics_for_iteration(dist, iteration, preds)
    assert m["distractor_retrieval_condition"] == 1.0
    assert abs(m["distractor_adoption_rate"] - 2 / 4) < 1e-9   # Claremont, Lebanon
    assert abs(m["gold_match_rate"] - 1 / 4) < 1e-9            # Newport
    assert m["distinct_answers"] == 4
    assert abs(m["offtarget_rate"] - 1 / 4) < 1e-9             # Keene only


# ── eval-side metric helper ───────────────────────────────────────────────────
def test_eval_injection_metrics():
    from hotpot_evaluation import _injection_metrics_for_iteration
    inj = {
        "mode": "counterfactual", "eligible": True, "gold_answer": "Newport",
        "injected_entity": "Claremont", "injected_doc_ids": ["gen_1_0"], "implied_answer": None,
    }
    iteration = {
        "documents": [{"doc_id": "gen_1_0", "text": "The seat is Claremont."},
                      {"doc_id": "corpus_5", "text": "unrelated"}],
        "runs": [],
    }
    preds = ["The seat is Claremont.", "It is Claremont", "The seat is Newport."]
    m = _injection_metrics_for_iteration(inj, iteration, preds)
    assert m["retrieval_condition"] == 1.0
    assert abs(m["injected_match_rate"] - 2 / 3) < 1e-9
    assert abs(m["gold_match_rate"] - 1 / 3) < 1e-9
    assert m["doc_propagation"] == 1.0
