"""T1 step 2: re-prompt each stored (question, round, context) triple with a DIRECT citation
elicitation prompt and record which documents the model says it used.

Backend-agnostic OpenAI-compatible client — set env vars, pick the backend at the command line:
  # vLLM server (no key):     OPENAI_BASE_URL=$VLLM_API_BASE  T1_MODEL=<served-name>
  # keymaker (LiteLLM proxy): OPENAI_BASE_URL=https://thekeymaker.umass.edu/v1  OPENAI_API_KEY=$API_KEY  T1_MODEL=<id>
  python t1_generate.py

NOTE ON THE PROMPT: the earlier "direct-elicitation" version's exact prompt is owned by Rati and
is not in this repo. The prompt below is a faithful reconstruction (answer using only the numbered
docs, then emit `CITED: [n,...]`). It doubles as the W2 appendix text and should be reconciled
with Rati's original wording before publication. The comparison statistic (over-citation ratio,
doc-level agreement vs LOO) is robust to exact wording; the wording only needs to be reported.

Output: camera_ready_outputs/T1/t1_direct.jsonl  {item_id, answer, cited_n:[...], cited_ids:[...]}
"""
import json, os, re, sys, time

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "camera_ready_outputs", "T1")
items = [json.loads(l) for l in open(os.path.join(OUT, "t1_sample.jsonl"), encoding="utf-8")]

MODEL = os.environ.get("T1_MODEL", "Qwen/Qwen2.5-14B-Instruct")
BASE = os.environ.get("OPENAI_BASE_URL") or os.environ.get("VLLM_API_BASE")
KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or "EMPTY"
if not BASE:
    sys.exit("set OPENAI_BASE_URL (vLLM $VLLM_API_BASE, or the keymaker /v1 URL)")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")
client = OpenAI(base_url=BASE, api_key=KEY)

SYSTEM = ("You are a helpful AI assistant that answers questions using only the information in the "
          "provided numbered context documents, and reports which documents you used.")

def user_prompt(question, docs):
    lines = [f"[{dc['n']}] {dc['text']}" for dc in docs]
    return ("Context documents:\n" + "\n".join(lines) +
            f"\n\nQuestion: {question}\n\n"
            "Answer the question using ONLY the information in the numbered documents above. "
            "Then, on a final separate line, list the numbers of the documents you actually used to "
            "support your answer, exactly in this format:\nCITED: [n, n, ...]\n"
            "If you used none, write CITED: [].")

CITED_RE = re.compile(r"CITED:\s*\[([0-9,\s]*)\]", re.I)

def parse_cited(text, docs):
    m = None
    for m in CITED_RE.finditer(text):
        pass  # take the last CITED: line
    if not m:
        return [], []
    nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
    by_n = {dc["n"]: dc["doc_id"] for dc in docs}
    ids = [by_n[n] for n in nums if n in by_n]
    return nums, ids

out_path = os.path.join(OUT, "t1_direct.jsonl")
done = set()
if os.path.exists(out_path):  # resume
    done = {json.loads(l)["item_id"] for l in open(out_path, encoding="utf-8")}
    print(f"resuming; {len(done)} already done")

with open(out_path, "a", encoding="utf-8") as fout:
    for k, it in enumerate(items):
        if it["item_id"] in done:
            continue
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=512,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user_prompt(it["question_text"], it["docs"])}],
            )
            ans = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [{it['item_id']}] error: {e}; retrying once"); time.sleep(2)
            resp = client.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=512,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user_prompt(it["question_text"], it["docs"])}])
            ans = resp.choices[0].message.content or ""
        cited_n, cited_ids = parse_cited(ans, it["docs"])
        fout.write(json.dumps(dict(item_id=it["item_id"], answer=ans, cited_n=cited_n,
                                   cited_ids=cited_ids), ensure_ascii=False) + "\n")
        fout.flush()
        if k % 25 == 0:
            print(f"  {k}/{len(items)}")
print("wrote", out_path)
