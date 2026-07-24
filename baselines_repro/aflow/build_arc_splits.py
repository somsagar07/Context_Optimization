"""Build AFlow-format JSONL datasets matched to ARC's eval protocol.

- validate (search) sets: sampled from HF *train* splits, seed 42, sizes matching
  AFlow's shipped validate sets (264 gsm8k / 200 hotpotqa / 200 drop) so the
  search budget is comparable to published AFlow runs.
- test sets: the FULL split ARC evaluates on (gsm8k test, hotpot_qa fullwiki
  validation, drop validation), in index order like ARC's eval_hrl.py.
- HotpotQA context is left empty: ARC's loader is question-only, so AFlow's
  workflows must answer question-only too (matched protocol).
- DROP input formatted exactly like ARC's drop_loader: "{passage}\n\nQuestion: {q}".
"""
import json, random
from datasets import load_dataset

OUT = "/data/ssagar6/NeurIPS_26/baselines/AFlow/data/datasets"
random.seed(42)

def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(path, len(rows))

# ---------------- GSM8K ----------------
gtrain = load_dataset("gsm8k", "main", split="train")
gtest = load_dataset("gsm8k", "main", split="test")
idx = random.sample(range(len(gtrain)), 264)
dump(f"{OUT}/gsm8k_validate.jsonl",
     [{"question": gtrain[i]["question"], "answer": gtrain[i]["answer"], "id": f"train_{i}"} for i in idx])
dump(f"{OUT}/gsm8k_test.jsonl",
     [{"question": r["question"], "answer": r["answer"], "id": f"test_{i}"} for i, r in enumerate(gtest)])

# ---------------- HotpotQA (fullwiki, question-only like ARC) ----------------
htrain = load_dataset("hotpot_qa", "fullwiki", split="train")
hval = load_dataset("hotpot_qa", "fullwiki", split="validation")
idx = random.sample(range(len(htrain)), 200)
def hrow(r, rid):
    return {"_id": rid, "question": r["question"], "answer": r["answer"],
            "context": [], "supporting_facts": [], "type": r.get("type", ""), "level": r.get("level", "")}
dump(f"{OUT}/hotpotqa_validate.jsonl", [hrow(htrain[i], f"train_{i}") for i in idx])
dump(f"{OUT}/hotpotqa_test.jsonl", [hrow(r, f"val_{i}") for i, r in enumerate(hval)])

# ---------------- DROP (ARC passage+question formatting) ----------------
dtrain = load_dataset("drop", split="train")
dval = load_dataset("drop", split="validation")
def drow(r, rid):
    q = f"{r['passage']}\n\nQuestion: {r['question']}"
    spans = r.get("answers_spans", {}).get("spans", [])
    ans = spans[0] if spans else ""
    return {"context": q, "ref_text": ans, "completion": "", "id": rid}
idx = random.sample(range(len(dtrain)), 200)
dump(f"{OUT}/drop_validate.jsonl", [drow(dtrain[i], f"train_{i}") for i in idx])
dump(f"{OUT}/drop_test.jsonl", [drow(r, f"val_{i}") for i, r in enumerate(dval)])
