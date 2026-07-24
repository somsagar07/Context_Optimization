"""Port DAAO to GAIA and MedQA (same recipe as the HotpotQA/DROP ports).

Self-contained: derives templates by COPYING the already-working HotpotQA port
(does NOT import/re-run the earlier builder, which would wipe trained
controllers). Scorers are taken verbatim from the AFlow benchmark files where
ARC's scorers were already transcribed.

- GAIA: ARC's custom split — train = first 65 of the seed-42-shuffled validation
  set (same 65 ARC's policy trained on), test = remaining 100.
- MedQA: 200 train-split questions (same as AFlow's search set), full test 1273.
"""
import json
import re
import shutil

ROOT = "/data/ssagar6/NeurIPS_26/baselines/DAAO"
OPT = f"{ROOT}/daao/ext/maas/scripts/optimized"
BENCH = f"{ROOT}/daao/ext/maas/benchmark"
DATA = f"{ROOT}/daao/ext/maas/data"
AFLOW_DATA = "/data/ssagar6/NeurIPS_26/baselines/AFlow/data/datasets"
AFLOW_BENCH = "/data/ssagar6/NeurIPS_26/baselines/AFlow/benchmarks"

def extract_scorer(path, fn_name):
    src = open(path).read()
    m = re.search(rf"(def {fn_name}\(.*?)(?=\n\nclass )", src, re.DOTALL)
    return m.group(1).rstrip().replace(f"def {fn_name}(", "def arc_evaluate_correctness(", 1)

SCORERS = {
    "GAIA": extract_scorer(f"{AFLOW_BENCH}/gaia.py", "arc_gaia_evaluate_correctness"),
    "MedQA": extract_scorer(f"{AFLOW_BENCH}/medqa.py", "arc_medqa_evaluate_correctness"),
}
EXTRA_IMPORTS = {"GAIA": "import string\nimport warnings\n", "MedQA": ""}

hotpot_bench = open(f"{BENCH}/hotpotqa.py").read()
m = re.search(r"(def arc_evaluate_correctness\(.*?)(?=\n\nclass )", hotpot_bench, re.DOTALL)
hotpot_scorer_block = m.group(1).rstrip()

def build_dataset(ds):
    # per-dataset workflow dirs from the working HotpotQA port (no round_* state)
    for phase in ["train", "test"]:
        dst = f"{OPT}/{ds}/{phase}"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(f"{OPT}/HotpotQA/{phase}", dst,
                        ignore=shutil.ignore_patterns("__pycache__", "round_*", "*.csv"))
        for fname in ["graph.py", "template/operator_registry.py"]:
            p = f"{dst}/{fname}"
            s = open(p).read().replace(f"optimized.HotpotQA.{phase}", f"optimized.{ds}.{phase}")
            open(p, "w").write(s)
        open(f"{dst}/results.json", "w").write("[]")

    # benchmark class = hotpotqa port with scorer + class name swapped
    code = hotpot_bench.replace(hotpot_scorer_block, SCORERS[ds])
    code = code.replace("class HotpotQABenchmark(", f"class {ds}Benchmark(")
    code = code.replace("from daao.logs import logger",
                        "from daao.logs import logger\n" + EXTRA_IMPORTS[ds])
    open(f"{BENCH}/{ds.lower()}.py", "w").write(code)
    print(f"built {ds}")

for ds in ["GAIA", "MedQA"]:
    build_dataset(ds)

# evaluator dispatch
p = f"{ROOT}/daao/ext/maas/scripts/evaluator.py"
s = open(p).read()
if "GAIABenchmark" not in s:
    s = s.replace("from daao.ext.maas.benchmark.drop import DROPBenchmark",
                  "from daao.ext.maas.benchmark.drop import DROPBenchmark\n"
                  "from daao.ext.maas.benchmark.gaia import GAIABenchmark\n"
                  "from daao.ext.maas.benchmark.medqa import MedQABenchmark")
    s = s.replace('"HotpotQA", "DROP"]', '"HotpotQA", "DROP", "GAIA", "MedQA"]')
    s = s.replace('"DROP": DROPBenchmark,',
                  '"DROP": DROPBenchmark,\n            "GAIA": GAIABenchmark,\n            "MedQA": MedQABenchmark,')
    open(p, "w").write(s)
    print("evaluator patched")

# experiment configs (same QA operator set as the other ports)
p = f"{ROOT}/daao/ext/maas/benchmark/experiment_configs.py"
s = open(p).read()
if '"GAIA"' not in s:
    mm = re.search(r'"HotpotQA": ExperimentConfig\(.*?operators=(\[[^\]]*\])', s, re.DOTALL)
    ops = mm.group(1)
    block = ""
    for ds in ["GAIA", "MedQA"]:
        block += (f'    "{ds}": ExperimentConfig(\n'
                  f'        dataset="{ds}",\n'
                  f'        question_type="qa",\n'
                  f'        operators={ops},\n    ),\n')
    s = s.replace('    "HumanEval": ExperimentConfig(', block + '    "HumanEval": ExperimentConfig(')
    open(p, "w").write(s)
    print("configs patched")

# data
for src, dst in [
    (f"{AFLOW_DATA}/gaia_validate.jsonl", f"{DATA}/gaia_train.jsonl"),
    (f"{AFLOW_DATA}/gaia_test.jsonl", f"{DATA}/gaia_test.jsonl"),
    (f"{AFLOW_DATA}/medqa_validate.jsonl", f"{DATA}/medqa_train.jsonl"),
    (f"{AFLOW_DATA}/medqa_test.jsonl", f"{DATA}/medqa_test.jsonl"),
]:
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            r = json.loads(line)
            out.write(json.dumps({"question": r["question"], "answer": r["answer"], "id": r.get("id", "")}) + "\n")
    print(dst, sum(1 for _ in open(dst)))

print("GAIA+MedQA PORT BUILD COMPLETE")
