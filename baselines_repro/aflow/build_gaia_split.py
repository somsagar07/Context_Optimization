"""Build AFlow GAIA JSONLs using ARC's own GAIADataset loader so the shuffle
(seed 42) and 65/rest train-eval partition are byte-identical to the paper.
Questions are formatted exactly as ARC's get_sample() does (including the
file-attachment system notification when present)."""
import importlib
import json
import os
import sys
import types

ARC = "/data/ssagar6/NeurIPS_26/Context_Optimization"
pkg = types.ModuleType("arc_utils"); pkg.__path__ = [f"{ARC}/utils"]
sub = types.ModuleType("arc_utils.data_loader"); sub.__path__ = [f"{ARC}/utils/data_loader"]
sys.modules["arc_utils"] = pkg
sys.modules["arc_utils.data_loader"] = sub
GAIADataset = importlib.import_module("arc_utils.data_loader.gaia_loader").GAIADataset

OUT = "/data/ssagar6/NeurIPS_26/baselines/AFlow/data/datasets"

def rows(ds):
    out = []
    for i in range(len(ds.data)):
        sample = ds.data[i]
        q = sample["Question"]
        rel = sample.get("file_path", "")
        if rel:
            full = os.path.join(ds.data_dir, rel)
            q += f"\n\n[System Notification]\nFile Attachment: {full}\nYou can use your tools to read or process this file."
        out.append({"question": q, "answer": sample["Final answer"], "id": f"gaia_{i}"})
    return out

train = GAIADataset(rl_split="train")
evald = GAIADataset(rl_split="eval")
for name, ds in [("gaia_validate.jsonl", train), ("gaia_test.jsonl", evald)]:
    rs = rows(ds)
    with open(f"{OUT}/{name}", "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
    print(name, len(rs))
