# Baseline Comparison Reproduction (Rebuttal E4 — AFlow & DAAO)

All results in `RESULTS.md`. Live working copies on the lab machine:
`/data/ssagar6/NeurIPS_26/baselines/{AFlow,DAAO}` (conda envs `aflow`, `daao`).
This directory contains everything needed to reproduce from upstream.

## Common protocol (both baselines)

- Executor for EVERY benchmark-answering call: `qwen/qwen-2.5-7b-instruct` via
  OpenRouter, temperature 0 — byte-identical to ARC's Table 1 backbone/serving.
- Test sets = ARC's exact eval splits (HF `test`; DROP/HotpotQA `validation`;
  GAIA = seed-42 shuffle of 2023_all validation, samples 65+ (n=100); MedQA
  `test`), built via `aflow/build_arc_splits.py`, `build_gaia_split.py` (uses
  ARC's own loader classes). Search/train sets = 264/200/200/65/200 problems
  from TRAIN splits (seed 42) — no test leakage.
- Scoring = ARC's `evaluate_correctness` per dataset (imported or copied
  verbatim; see `aflow/rescore_arc_metrics.py` and the benchmark files here).
- HotpotQA is question-only (ARC protocol) — AFlow's native setting feeds gold
  context; do NOT use their shipped hotpotqa files.
- Mean±std = 3 repeated test evals of the frozen artifact (no retraining).

## AFlow  (upstream FoundationAgents/AFlow @ 3f457218)

1. Clone upstream, `conda env` py3.9, `pip install -r requirements.txt` + requests.
2. `git apply aflow_modifications.patch` (GAIA/MedQA registration in
   run.py/evaluator.py). Add `benchmarks/gaia.py`, `benchmarks/medqa.py` from here.
3. `config/config2.yaml` from `config2.yaml.template` (insert key).
4. Build datasets: `python build_arc_splits.py && python build_gaia_split.py`
   (+ MedQA builder — see RESULTS.md; formats are 1-line JSONL question/answer).
5. Search (per dataset): `python run.py --dataset GSM8K --max_rounds 20
   --opt_model_name anthropic/claude-sonnet-5 --exec_model_name qwen/qwen-2.5-7b-instruct`
   - MUST run in the default `workspace/` dir (generated code hardcodes
     `workspace.` imports); seed each `workflows/results.json` with `[]`.
   - Budget-matched variant: `--opt_model_name qwen/qwen-2.5-7b-instruct`.
6. Test best round: `python run_test_eval.py <DATASET> <ROUND>`; re-score with
   `python rescore_arc_metrics.py <csv> <dataset>` (base env with `datasets`).
   Best rounds used: GSM8K 5, HotpotQA 7, DROP 15, GAIA 4, MedQA 1 (sonnet);
   GSM8K 12, HotpotQA 15, DROP 1, GAIA 1, MedQA 1 (qwen-opt).

## DAAO  (upstream AutoAgents-ai/DAAO @ 5e260bb9)

1. Clone upstream, conda py3.10. Install needs fixes (in the patch):
   requirements.txt has CRLF line endings, `typing_extensions==4.9.0` conflicts
   with its own altair pin (→4.12.2), torch pins need
   `--extra-index-url https://download.pytorch.org/whl/cu118`. Then `pip install -e .`.
2. `git apply daao_modifications.patch`. Released-code fixes included (all in
   DAAO's favor; disclose in paper):
   - train graph: `.item()` detached the policy log-prob → controller could not
     train (also crashed torch.stack in the loss). Kept as tensor.
   - GSM8K test registry imported `EarlyStop` from the train template where it
     is commented out → import from test template.
   - shipped GSM8K test graph incompatible with trained controller signature →
     evaluate with the train-time graph (test-template imports).
   - NOTE: released trainer does ONE repetition regardless of the 4-rep setting
     (`return` inside the loop) — left as released; README protocol followed.
3. Ports (our extension, disclosed): run `python build_daao_qa_ports.py` then
   `python build_daao_gaia_medqa.py` (creates HotpotQA/DROP/GAIA/MedQA benchmark
   classes + per-dataset workflow stacks mirroring their GSM8K stack minus the
   math-only Programmer operator; ARC scorers; QA prompt requires
   "Final Answer: <letter>: <full option text>" on MedQA — bare letters are
   unscorable against ARC's text ground truth and deflate DAAO unfairly).
4. `config/config2.yaml` from template (router restricted to the qwen executor).
5. Train + test per dataset:
   `python examples/maas/optimize.py --dataset <DS> --opt_model_name openai/gpt-4o-mini
    --exec_model_name qwen/qwen-2.5-7b-instruct --sample 4 --round 1 --batch_size 4 --lr 0.01`
   then the same with `--is_test`.

## Cost notes

Search+train+test for everything above ran ~$15 total at qwen rates
($0.04/$0.10 per M) + ~$0.5 claude-sonnet-5 (AFlow optimizer, 49 calls) +
~$1 gpt-4o-mini (DAAO optimizer). Repeat evals ~$12.
