"""Port DAAO to HotpotQA and DROP (ARC rebuttal).

Mirrors DAAO's GSM8K stack as closely as possible:
- benchmark classes copy GSM8K's VAE/logprob plumbing verbatim, with scoring
  swapped to ARC's evaluate_correctness logic (HotpotQA containment; DROP
  number/date/span/list matcher) copied from ARC's loaders.
- operator set = all non-code operators from their GSM8K registry
  (Generate, GenerateCoT, MultiGenerateCoT, ScEnsemble, SelfRefine), with the
  config list and registry kept CONSISTENT (their GSM8K release lists 7 ops in
  the config but registers 6 — a latent index-out-of-range we avoid here).
- graph.py = their GSM8K graph with the math prompt swapped for a QA prompt and
  the final Programmer (python) verification step removed (QA answers are not
  numeric; code verification is a math-specific step).
- train/test templates mirrored like GSM8K (test registry imports test template).
"""
import json
import os
import re
import shutil

ROOT = "/data/ssagar6/NeurIPS_26/baselines/DAAO"
OPT = f"{ROOT}/daao/ext/maas/scripts/optimized"
BENCH = f"{ROOT}/daao/ext/maas/benchmark"
DATA = f"{ROOT}/daao/ext/maas/data"
AFLOW_DATA = "/data/ssagar6/NeurIPS_26/baselines/AFlow/data/datasets"

QA_OPS = ["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "SelfRefine"]

# ---------------------------------------------------------------- scorers
HOTPOT_SCORER = '''
def arc_evaluate_correctness(prediction: str, ground_truth: str) -> float:
    """ARC HotpotQA scorer (bidirectional containment, case-insensitive)."""
    pred = str(prediction).lower().strip()
    truth = str(ground_truth).lower().strip()
    if not pred:
        return 0.0
    if truth in pred or pred in truth:
        return 1.0
    return 0.0
'''

DROP_SCORER = '''
def _normalize_number(text):
    text = text.replace(',', '').replace('$', '').strip()
    nums = re.findall(r"[-+]?\\d*\\.\\d+|\\d+", text)
    if nums:
        try:
            return float(nums[0])
        except ValueError:
            return None
    return None

def _normalize_date(text):
    text = re.sub(r'\\b(on|the|of)\\b', '', text, flags=re.IGNORECASE)
    return text.strip()

def _extract_answer_from_prediction(prediction):
    answer_patterns = [
        r'Final Answer[:\\s]+(.+?)(?:\\.|$|\\n)',
        r'Answer[:\\s]+(.+?)(?:\\.|$|\\n)',
        r'####\\s*(.+)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, prediction, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    if '"' in prediction:
        quoted = re.findall(r'"([^"]+)"', prediction)
        if quoted:
            return quoted[-1]
    lines = [l.strip() for l in prediction.split('\\n') if l.strip()]
    if lines:
        return lines[-1]
    return prediction.strip()

def arc_evaluate_correctness(prediction: str, ground_truth: str) -> float:
    """ARC DROP scorer (numbers, dates, spans, lists) — copied from drop_loader."""
    pred_answer = _extract_answer_from_prediction(str(prediction)).strip()
    ground_truth = str(ground_truth).strip()

    pred_num = _normalize_number(pred_answer)
    truth_num = _normalize_number(ground_truth)
    if pred_num is not None and truth_num is not None:
        return 1.0 if abs(pred_num - truth_num) < 1e-3 else 0.0

    pred_date = _normalize_date(pred_answer)
    truth_date = _normalize_date(ground_truth)
    if pred_date and truth_date:
        if pred_date.lower() == truth_date.lower():
            return 1.0

    pred_lower = pred_answer.lower()
    truth_lower = ground_truth.lower()
    pred_clean = re.sub(r'[^\\w\\s]', '', pred_lower)
    truth_clean = re.sub(r'[^\\w\\s]', '', truth_lower)
    if pred_clean == truth_clean:
        return 1.0
    if truth_clean:
        if truth_clean in pred_clean:
            return 1.0
        truth_words = set(truth_clean.split())
        pred_words = set(pred_clean.split())
        if truth_words and truth_words.issubset(pred_words):
            return 1.0
        if len(truth_words) == 1:
            word = list(truth_words)[0]
            pattern = r'\\b' + re.escape(word) + r'\\b'
            if re.search(pattern, pred_clean):
                return 1.0
    if ',' in ground_truth or ';' in ground_truth:
        truth_parts = [t.strip() for t in re.split(r'[,;]', ground_truth)]
        for part in truth_parts:
            part_clean = re.sub(r'[^\\w\\s]', '', part.lower())
            if part_clean:
                if part_clean == pred_clean:
                    return 1.0
                if part_clean in pred_clean or pred_clean in part_clean:
                    return 1.0
                part_words = set(part_clean.split())
                pred_words = set(pred_clean.split())
                if part_words and part_words.issubset(pred_words):
                    return 1.0
    return 0.0
'''

BENCH_TEMPLATE = '''import asyncio
import torch
import re
from typing import Callable, List

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from daao.ext.maas.benchmark.benchmark import BaseBenchmark
from daao.logs import logger

{scorer}

class {cls}(BaseBenchmark):
    def __init__(self,
                name: str,
                file_path: str,
                log_path: str,
                batch_size: int,
                controller: torch.nn.Module,
                operator_embeddings: List[List[float]],
                optimizer: torch.optim.Optimizer,):
        super().__init__(name, file_path, log_path, batch_size, controller, operator_embeddings, optimizer)

    @retry(stop=stop_after_attempt(20), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await asyncio.wait_for(graph(input_text), timeout=1500)

    async def evaluate_problem(self, problem: dict, graph: Callable):
        input_text = problem["question"]
        expected_output = problem["answer"]

        try:
            output, cost, logprob, vae = await self._generate_output(graph, input_text)
            if not output:
                raise ValueError("output is empty")

            score = arc_evaluate_correctness(output, expected_output)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, output)
                vae["is_solved"] = 0
            else:
                vae["is_solved"] = 1

            return input_text, output, expected_output, score, cost, logprob, vae

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {{e}}")
            vae = {{
                    "z_difficulty": torch.zeros((1, 32), device=self.device),
                   "difficulty_scalar": torch.tensor(0.5, device=self.device),
                   "mu": torch.zeros((1, 32), device=self.device),
                   "logvar": torch.zeros((1, 32), device=self.device),
                   "is_solved": 0
            }}
            return input_text, str(e), expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device), vae

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob", "vae"]
'''

QA_PROMPT = '''

QA_SOLVE_PROMPT = """
Answer the following question. Reason step by step if helpful. End your response
with your final answer on a new line in the format: Final Answer: <answer>
"""
'''

OPERATOR_JSON = {
    "Generate": {
        "description": "Generates anything based on customized input and instruction.",
        "interface": "generate(input: str, instruction: str) -> dict with key 'response' of type str"
    },
    "GenerateCoT": {
        "description": "Generates an answer using a chain-of-thought approach, providing step-by-step reasoning before producing the final answer. Useful for multi-hop or reading-comprehension questions.",
        "interface": "generate_cot(input: str, instruction: str) -> dict with key 'response' of type str"
    },
    "MultiGenerateCoT": {
        "description": "Generates multiple answers using diverse chain-of-thought reasoning processes to increase answer variety and robustness.",
        "interface": "multi_generate_cot(input: str, instruction: str) -> dict with key 'response' of type List[str]"
    },
    "ScEnsemble": {
        "description": "Uses self-consistency to select the answer that appears most frequently in the candidate list.",
        "interface": "sc_ensemble(solutions: List[str], problem: str) -> dict with key 'response' of type str"
    },
    "SelfRefine": {
        "description": "Refines the generated answer by analyzing errors or unsupported claims and making iterative improvements.",
        "interface": "self_refine(problem: str, solution: str) -> dict with key 'response' of type str"
    }
}

def make_registry(ds, phase):
    ops = ",\n    ".join(QA_OPS)
    mapping = ",\n    ".join(f'"{o}": {o}' for o in QA_OPS)
    return (f"from daao.ext.maas.scripts.optimized.{ds}.{phase}.template.operator import (\n"
            f"    {ops},\n)\n\n"
            f"operator_mapping = {{\n    {mapping},\n}}\n\n"
            f"operator_names = list(operator_mapping.keys())\n")

def build_dataset(ds):
    for phase in ["train", "test"]:
        dst = f"{OPT}/{ds}/{phase}"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(f"{OPT}/GSM8K/train", dst,
                        ignore=shutil.ignore_patterns("__pycache__", "round_*", "*.csv", "results.json"))
        # graph.py: retarget imports at this dataset/phase's template
        gp = f"{dst}/graph.py"
        g = open(gp).read()
        g = g.replace("optimized.GSM8K.train.template", f"optimized.{ds}.{phase}.template")
        g = g.replace("prompt_custom.MATH_SOLVE_PROMPT", "prompt_custom.QA_SOLVE_PROMPT")
        # remove the math-specific Programmer verification tail; return final answer
        g = re.sub(
            r"        verification = await self\.programmer.*?return final_solution, total_cost, sum_log_prob, vae",
            "        for key, value in llm_instance.items():\n"
            "            total_cost += value.cost_manager.total_cost\n\n"
            "        return final_solution, total_cost, sum_log_prob, vae",
            g, flags=re.DOTALL)
        assert "self.programmer(" not in g.split("def __call__")[1]
        open(gp, "w").write(g)
        # registry: consistent 5-op QA set
        open(f"{dst}/template/operator_registry.py", "w").write(make_registry(ds, phase))
        # prompt: append QA prompt
        with open(f"{dst}/template/prompt.py", "a") as f:
            f.write(QA_PROMPT)
        # operator.json: QA descriptions
        json.dump(OPERATOR_JSON, open(f"{dst}/template/operator.json", "w"), indent=4)
        os.makedirs(f"{dst}", exist_ok=True)
        open(f"{dst}/results.json", "w").write("[]")

    # benchmark class
    scorer = HOTPOT_SCORER if ds == "HotpotQA" else DROP_SCORER
    cls = f"{ds}Benchmark"
    open(f"{BENCH}/{ds.lower()}.py", "w").write(BENCH_TEMPLATE.format(scorer=scorer, cls=cls))

for ds in ["HotpotQA", "DROP"]:
    build_dataset(ds)

# ---- evaluator patch
p = f"{ROOT}/daao/ext/maas/scripts/evaluator.py"
s = open(p).read()
if "HotpotQABenchmark" not in s:
    s = s.replace("from daao.ext.maas.benchmark.math import MATHBenchmark",
                  "from daao.ext.maas.benchmark.math import MATHBenchmark\n"
                  "from daao.ext.maas.benchmark.hotpotqa import HotpotQABenchmark\n"
                  "from daao.ext.maas.benchmark.drop import DROPBenchmark")
    s = s.replace('DatasetType = Literal["HumanEval", "GSM8K", "MATH"]',
                  'DatasetType = Literal["HumanEval", "GSM8K", "MATH", "HotpotQA", "DROP"]')
    s = s.replace('"HumanEval": HumanEvalBenchmark,',
                  '"HumanEval": HumanEvalBenchmark,\n            "HotpotQA": HotpotQABenchmark,\n            "DROP": DROPBenchmark,')
    open(p, "w").write(s)

# ---- experiment configs patch
p = f"{ROOT}/daao/ext/maas/benchmark/experiment_configs.py"
s = open(p).read()
if '"HotpotQA"' not in s:
    ops = json.dumps(QA_OPS)
    block = ""
    for ds in ["HotpotQA", "DROP"]:
        block += (f'    "{ds}": ExperimentConfig(\n'
                  f'        dataset="{ds}",\n'
                  f'        question_type="qa",\n'
                  f'        operators={ops},\n    ),\n')
    s = s.replace('    "HumanEval": ExperimentConfig(', block + '    "HumanEval": ExperimentConfig(')
    open(p, "w").write(s)

# ---- data files (ARC-matched, from the AFlow builds)
for src, dst, conv in [
    (f"{AFLOW_DATA}/hotpotqa_validate.jsonl", f"{DATA}/hotpotqa_train.jsonl", "hotpot"),
    (f"{AFLOW_DATA}/hotpotqa_test.jsonl", f"{DATA}/hotpotqa_test.jsonl", "hotpot"),
    (f"{AFLOW_DATA}/drop_validate.jsonl", f"{DATA}/drop_train.jsonl", "drop"),
    (f"{AFLOW_DATA}/drop_test.jsonl", f"{DATA}/drop_test.jsonl", "drop"),
]:
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            r = json.loads(line)
            if conv == "hotpot":
                row = {"question": r["question"], "answer": r["answer"], "id": r.get("_id", "")}
            else:
                row = {"question": r["context"], "answer": r["ref_text"], "id": r.get("id", "")}
            out.write(json.dumps(row) + "\n")
    print(dst, sum(1 for _ in open(dst)))

print("PORT BUILD COMPLETE")
