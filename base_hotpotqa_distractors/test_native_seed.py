"""Pure-Python tests (no GPU) for the ORIGINAL HotpotQA distractor-setting round-0 seed
(pipeline.misinfo.native_context_docs). Run from the repo root:

    /c/Users/riddh/anaconda3/envs/rag-collapse/python.exe -m pytest base_hotpotqa_distractors/ -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from pipeline.misinfo import native_context_docs

REC = {
    "_id": "q1", "answer": "Providence",
    "supporting_facts": [["Brown University", 0], ["Rhode Island", 1]],
    "context": [
        ["Brown University", ["Brown University is in Providence.", "Founded in 1764."]],
        ["Rhode Island", ["Providence is the capital of Rhode Island."]],
        ["Newport", ["Newport is a city in Rhode Island."]],
        ["Boston", ["Boston is the capital of Massachusetts."]],
    ],
}


def test_gold_distractor_split():
    docs = native_context_docs(REC)
    assert len(docs) == 4
    gold = [d for d in docs if d["gold"]]
    distract = [d for d in docs if d["distractor"]]
    assert len(gold) == 2 and len(distract) == 2
    assert {d["title"] for d in gold} == {"Brown University", "Rhode Island"}
    # gold and distractor are complementary tags
    assert all(d["gold"] != d["distractor"] for d in docs)


def test_doc_shape_and_sentence_join():
    docs = native_context_docs(REC)
    d0 = docs[0]
    assert d0["doc_id"] == "native_0" and d0["iteration"] == 0 and d0["url"] == ""
    assert d0["text"] == "Brown University is in Providence. Founded in 1764."  # sentences joined


def test_gold_only_drops_distractors():
    docs = native_context_docs(REC, gold_only=True)
    assert len(docs) == 2 and all(d["gold"] for d in docs)
    assert {d["title"] for d in docs} == {"Brown University", "Rhode Island"}


def test_empty_context():
    assert native_context_docs({"_id": "x", "supporting_facts": [], "context": []}) == []


def test_all_distractor_when_no_supporting_titles_match():
    rec = {"_id": "y", "answer": "Z", "supporting_facts": [["Missing Title", 0]],
           "context": [["A", ["a."]], ["B", ["b."]]]}
    docs = native_context_docs(rec)
    assert len(docs) == 2 and all(d["distractor"] for d in docs)
    assert native_context_docs(rec, gold_only=True) == []  # no gold paragraphs present


# --- fetch_distractor_file._convert: HF example (parallel lists) -> raw HotpotQA record ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # base_hotpotqa_distractors/
from fetch_distractor_file import _convert


def test_convert_hf_example():
    hf = {
        "id": "abc", "question": "Q?", "answer": "Providence", "type": "bridge", "level": "hard",
        "supporting_facts": {"title": ["Brown University", "Rhode Island"], "sent_id": [0, 1]},
        "context": {
            "title": ["Brown University", "Rhode Island", "Newport"],
            "sentences": [["Brown is in Providence.", "Founded 1764."],
                          ["Providence is the capital."], ["Newport is in RI."]],
        },
    }
    rec = _convert(hf)
    assert rec["_id"] == "abc" and rec["answer"] == "Providence" and rec["type"] == "bridge"
    assert rec["supporting_facts"] == [["Brown University", 0], ["Rhode Island", 1]]
    assert rec["context"] == [
        ["Brown University", ["Brown is in Providence.", "Founded 1764."]],
        ["Rhode Island", ["Providence is the capital."]],
        ["Newport", ["Newport is in RI."]],
    ]
    # round-trips through the seeder: 2 gold (supporting titles) + 1 distractor
    docs = native_context_docs(rec)
    assert len(docs) == 3 and sum(d["gold"] for d in docs) == 2 and sum(d["distractor"] for d in docs) == 1
