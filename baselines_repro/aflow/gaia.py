# -*- coding: utf-8 -*-
# GAIA benchmark for AFlow, scored with ARC's exact evaluate_correctness logic
# (copied verbatim from Context_Optimization/utils/data_loader/gaia_loader.py —
# GAIA official quasi-exact-match; kept dependency-free for the aflow env).
import re
import string
import warnings
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


def arc_gaia_evaluate_correctness(model_answer: str, ground_truth: str) -> float:
    match = re.search(r"Final Answer:\s*(.*)", model_answer, re.IGNORECASE | re.DOTALL)
    if match:
        model_answer = match.group(1).strip()
    else:
        model_answer = model_answer.strip()

    def _normalize_number_str(number_str: str) -> float:
        for char in ["$", "%", ","]:
            number_str = number_str.replace(char, "")
        try:
            return float(number_str)
        except ValueError:
            return float("inf")

    def _split_string(s: str, char_list: List[str] = [",", ";"]) -> List[str]:
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)

    def _is_float(element) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    def _normalize_str(input_str, remove_punct=True) -> str:
        no_spaces = re.sub(r"\s", "", input_str)
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        else:
            return no_spaces.lower()

    if model_answer is None:
        model_answer = "None"

    if _is_float(ground_truth):
        normalized_answer = _normalize_number_str(model_answer)
        return 1.0 if normalized_answer == float(ground_truth) else 0.0
    elif any(char in ground_truth for char in [",", ";"]):
        gt_elems = _split_string(ground_truth)
        ma_elems = _split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            warnings.warn("Answer lists have different lengths, returning False.", UserWarning)
            return 0.0
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if _is_float(gt_elem):
                normalized_ma_elem = _normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                comparisons.append(
                    _normalize_str(ma_elem, remove_punct=False)
                    == _normalize_str(gt_elem, remove_punct=False)
                )
        return 1.0 if all(comparisons) else 0.0
    else:
        return 1.0 if _normalize_str(model_answer) == _normalize_str(ground_truth) else 0.0


class GAIABenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        return arc_gaia_evaluate_correctness(str(prediction), str(expected_output)), prediction

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
