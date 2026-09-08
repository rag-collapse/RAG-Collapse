"""
Step 1 of making the quality-controlled regression use QWEN data.

Extracts Qwen round-1 references from all_experiments into the schema expected by
the LLM-judge scorer (step 2) and, after scoring, by provenance_regression.py.
Each record carries question, ref provenance (is_ai = self-generated), url (for the
GPTZero split), citation_rate, and the reference TEXT to be scored. No API needed.

Output: qwen_round1_to_score.jsonl  (one record per round-1 reference)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Qwen2.5-14B Replace-One raw run. Override with argv[1] or $QWEN_REPLACE_ONE_RUN.
# On Unity this lives under /work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments/.
QF = (sys.argv[1] if len(sys.argv) > 1 else
      os.environ.get("QWEN_REPLACE_ONE_RUN",
                     "/work/pi_dagarwal_umass_edu/project_4/file_storage/all_experiments/graphite/baseline/replace_one/experiment_outputs/Qwen/Qwen2.5-14B-Instruct/local_replace_one.json"))
OUT = os.path.join(HERE, "qwen_round1_to_score.jsonl")
ROUND = 1


def main():
    d = json.load(open(QF))
    n = 0
    with open(OUT, "w") as out:
        for q in d["questions"]:
            it = next((x for x in q["iterations"] if x["iteration_number"] == ROUND), None)
            if it is None:
                continue
            # citation counts this round, aggregated over runs
            cc, tot = {}, 0
            for r in it["runs"]:
                for c in (r.get("citations") or []):
                    cc[c["doc_id"]] = cc.get(c["doc_id"], 0) + 1
                    tot += 1
            if tot == 0:
                continue
            for doc in it["documents"]:
                did = doc["doc_id"]
                rec = {
                    "question": q["question_text"],
                    "question_id": q["question_id"],
                    "round": ROUND,
                    "doc_id": did,
                    "is_ai": str(did).startswith("gen_"),   # self-generated == "is_ai" in the regression
                    "url": doc.get("url", ""),
                    "citations": cc.get(did, 0),
                    "total_citations": tot,
                    "citation_rate": cc.get(did, 0) / tot,
                    "chunk": doc.get("text", ""),            # text to be scored in step 2
                }
                out.write(json.dumps(rec) + "\n")
                n += 1
    print(f"Wrote {n} round-{ROUND} references to {OUT}")


if __name__ == "__main__":
    main()
