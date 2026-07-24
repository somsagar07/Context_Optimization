"""Evaluate a searched AFlow workflow round on the full test set.

Usage: python run_test_eval.py <DATASET> <ROUND>
Uses data/datasets/<dataset>_test.jsonl (ARC-matched full eval split) and the
same executor model config as the search. Saves per-problem predictions CSV
under workspace/<DATASET>/test_round_<ROUND>/ for ARC-metric re-scoring.
"""
import asyncio
import os
import sys

from scripts.async_llm import LLMsConfig
from scripts.evaluator import Evaluator
from scripts.optimizer_utils.graph_utils import GraphUtils

EXEC_MODEL = "qwen/qwen-2.5-7b-instruct"

def main():
    dataset, rnd = sys.argv[1], int(sys.argv[2])
    exec_cfg = LLMsConfig.default().get(EXEC_MODEL)
    out_dir = f"workspace/{dataset}/test_round_{rnd}"
    os.makedirs(out_dir, exist_ok=True)

    gu = GraphUtils(f"workspace/{dataset}")
    graph_class = gu.load_graph(rnd, f"workspace/{dataset}/workflows")

    ev = Evaluator(eval_path=out_dir)
    score, avg_cost, total_cost = asyncio.run(
        ev.graph_evaluate(
            dataset,
            graph_class,
            {"dataset": dataset, "llm_config": exec_cfg},
            out_dir,
            is_test=True,
        )
    )
    print(f"TEST_RESULT dataset={dataset} round={rnd} score={score:.5f} "
          f"avg_cost={avg_cost:.6f} total_cost={total_cost:.4f}")

if __name__ == "__main__":
    main()
