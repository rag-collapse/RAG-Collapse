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
