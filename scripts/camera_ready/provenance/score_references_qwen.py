"""
Step 2: score the Qwen round-1 references on the 8 quality dimensions with a
Claude judge, producing a file provenance_regression.py can consume directly.

Input :  qwen_round1_to_score.jsonl   (from build_qwen_scoring_input.py)
Output:  qwen_round1_scored.jsonl     (adds the 8 dimension scores)

Run (needs ANTHROPIC_API_KEY in env):
    uv run --with anthropic python score_references_qwen.py
Then:
    uv run --with numpy python provenance_regression.py qwen_round1_scored.jsonl

Scores the first N_QUESTIONS questions (~8 refs each). Resumable: re-running skips
references already in the output. Judge is independent of the Qwen generators.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

JUDGE_MODEL = "claude-sonnet-4-6"   # switch to claude-opus-4-8 for a stronger judge
N_QUESTIONS = 400   # all questions with round-1 citations (378 of the 400)
HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "qwen_round1_to_score.jsonl")
OUT = os.path.join(HERE, "qwen_round1_scored.jsonl")

DIMENSIONS = ["overall", "relevance", "accuracy", "thoroughness",
              "specificity", "up_to_date", "organization", "direct_answer"]

SYSTEM = "You are evaluating how well a reference document answers a given question."

PROMPT = """A user asked the following question:

{question}

Below is a reference document that could be used to help answer the question.

--- Reference ---
{chunk}

Score this reference on each dimension on a 1-5 scale:
- overall: how good a source is this for answering the question? (1 not useful .. 5 excellent)
- relevance: how directly does it address the question? (1 off-topic .. 5 directly addresses)
- accuracy: how accurate/precise/internally consistent? (1 clear errors .. 5 precise and correct)
- thoroughness: how comprehensively does it cover the topic? (1 superficial .. 5 comprehensive)
- specificity: concrete facts/names/dates/numbers? (1 vague .. 5 highly specific)
- up_to_date: how current is the information? (1 very outdated .. 5 up to date)
- organization: clear structure/logical flow? (1 disorganized .. 5 excellent structure)
- direct_answer: how directly/explicitly does it state the answer? (1 scattered .. 5 explicitly states it)

Respond with ONLY a JSON object, no prose, with integer values 1-5 for exactly these keys:
overall, relevance, accuracy, thoroughness, specificity, up_to_date, organization, direct_answer"""


def score_one(client, rec):
    msg = PROMPT.format(question=rec["question"], chunk=rec["chunk"][:6000])
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=JUDGE_MODEL, max_tokens=300, system=SYSTEM,
                messages=[{"role": "user", "content": msg}],
            )
            text = resp.content[0].text
            m = re.search(r"\{.*\}", text, re.DOTALL)
            scores = json.loads(m.group(0))
            out = {k: rec[k] for k in ("question", "question_id", "round", "doc_id",
                                       "is_ai", "url", "citations", "total_citations", "citation_rate")}
            for d in DIMENSIONS:
                out[d] = int(scores[d])
            return out
        except Exception as e:
            if attempt == 3:
                print(f"  FAILED {rec['doc_id']} q{rec['question_id']}: {e}", file=sys.stderr)
                return None
    return None


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY first.")
    recs = [json.loads(l) for l in open(IN)]
    qids = sorted({r["question_id"] for r in recs})[:N_QUESTIONS]
    qset = set(qids)
    recs = [r for r in recs if r["question_id"] in qset]

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            done.add((r["question_id"], r["doc_id"]))
    todo = [r for r in recs if (r["question_id"], r["doc_id"]) not in done]
    print(f"Scoring {len(todo)} refs across {len(qids)} questions with {JUDGE_MODEL} "
          f"({len(done)} already done)")

    client = anthropic.Anthropic()
    n = 0
    with open(OUT, "a") as out, ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(score_one, client, r): r for r in todo}
        for f in as_completed(futs):
            res = f.result()
            if res:
                out.write(json.dumps(res) + "\n")
                out.flush()
                n += 1
                if n % 50 == 0:
                    print(f"  {n}/{len(todo)}")
    print(f"Wrote {n} scored refs to {OUT}")


if __name__ == "__main__":
    main()
