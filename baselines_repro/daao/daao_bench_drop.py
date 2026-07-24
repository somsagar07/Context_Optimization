import asyncio
import torch
import re
from typing import Callable, List

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from daao.ext.maas.benchmark.benchmark import BaseBenchmark
from daao.logs import logger


def _normalize_number(text):
    text = text.replace(',', '').replace('$', '').strip()
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if nums:
        try:
            return float(nums[0])
        except ValueError:
            return None
    return None

def _normalize_date(text):
    text = re.sub(r'\b(on|the|of)\b', '', text, flags=re.IGNORECASE)
    return text.strip()

def _extract_answer_from_prediction(prediction):
    answer_patterns = [
        r'Final Answer[:\s]+(.+?)(?:\.|$|\n)',
        r'Answer[:\s]+(.+?)(?:\.|$|\n)',
        r'####\s*(.+)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, prediction, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    if '"' in prediction:
        quoted = re.findall(r'"([^"]+)"', prediction)
        if quoted:
            return quoted[-1]
    lines = [l.strip() for l in prediction.split('\n') if l.strip()]
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
    pred_clean = re.sub(r'[^\w\s]', '', pred_lower)
    truth_clean = re.sub(r'[^\w\s]', '', truth_lower)
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
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, pred_clean):
                return 1.0
    if ',' in ground_truth or ';' in ground_truth:
        truth_parts = [t.strip() for t in re.split(r'[,;]', ground_truth)]
        for part in truth_parts:
            part_clean = re.sub(r'[^\w\s]', '', part.lower())
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


class DROPBenchmark(BaseBenchmark):
    def __init__(self,
                name: str,
                file_path: str,
                log_path: str,
                batch_size: int,
                controller: torch.nn.Module,
                operator_embeddings: List[List[float]],
                optimizer: torch.optim.Optimizer,):
        super().__init__(name, file_path, log_path, batch_size, controller, operator_embeddings, optimizer)

    def calculate_score(self, expected_output, prediction):
        return arc_evaluate_correctness(str(prediction), str(expected_output)), prediction

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
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            vae = {
                    "z_difficulty": torch.zeros((1, 32), device=self.device),
                   "difficulty_scalar": torch.tensor(0.5, device=self.device),
                   "mu": torch.zeros((1, 32), device=self.device),
                   "logvar": torch.zeros((1, 32), device=self.device),
                   "is_solved": 0
            }
            return input_text, str(e), expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device), vae

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob", "vae"]
