# -*- coding: utf-8 -*-
# MedQA benchmark for AFlow, scored with ARC's exact evaluate_correctness logic
# (copied verbatim from Context_Optimization/utils/data_loader/medqa_loader.py;
# ground truth is the correct-answer TEXT, matching ARC's eval protocol).
import re
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


def arc_medqa_evaluate_correctness(prediction: str, ground_truth: str) -> float:
    pred_text = prediction.strip()

    gt_letter_match = re.match(r'^([A-D])\b', ground_truth)
    gt_letter = gt_letter_match.group(1) if gt_letter_match else None

    gt_text_normalized = re.sub(r'\W+', '', ground_truth.lower())
    if gt_letter:
        gt_text_normalized = re.sub(r'\W+', '', ground_truth[1:].lower())

    final_answer_patterns = [
        r"Final Answer\s*[:\-\s](.*)",
        r"The answer is\s*[:\-\s](.*)",
        r"Answer\s*[:\-\s](.*)"
    ]

    extracted_answer = pred_text
    for pattern in final_answer_patterns:
        match = re.search(pattern, pred_text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted_answer = match.group(1).strip()
            if "Final Answer" in pred_text:
                extracted_answer = pred_text.split("Final Answer")[-1]
            break

    if gt_letter:
        pred_letter_match = re.search(r'\b([A-D])\b', extracted_answer)
        if pred_letter_match:
            predicted_letter = pred_letter_match.group(1)
            return 1.0 if predicted_letter == gt_letter else 0.0

    pred_normalized = re.sub(r'\W+', '', extracted_answer.lower())

    if gt_text_normalized and gt_text_normalized in pred_normalized:
        return 1.0

    return 0.0


class MedQABenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        return arc_medqa_evaluate_correctness(str(prediction), str(expected_output)), prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = problem["question"]
        expected_output = problem["answer"]
        try:
            output, cost = await self._generate_output(graph, input_text)
            score, extracted_output = self.calculate_score(expected_output, output)
            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)
            return input_text, output, expected_output, score, cost
        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]
