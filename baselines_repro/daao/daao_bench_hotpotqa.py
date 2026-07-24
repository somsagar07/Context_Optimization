import asyncio
import torch
import re
from typing import Callable, List

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from daao.ext.maas.benchmark.benchmark import BaseBenchmark
from daao.logs import logger


def arc_evaluate_correctness(prediction: str, ground_truth: str) -> float:
    """ARC HotpotQA scorer (bidirectional containment, case-insensitive)."""
    pred = str(prediction).lower().strip()
    truth = str(ground_truth).lower().strip()
    if not pred:
        return 0.0
    if truth in pred or pred in truth:
        return 1.0
    return 0.0


class HotpotQABenchmark(BaseBenchmark):
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
