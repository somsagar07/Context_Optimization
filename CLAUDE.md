# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: ARC — Learning to Configure Agents

Hierarchical RL system that trains **two policies** to configure multi-agent LLM workflows:
- **Structure policy** (high-level, single-step, `MultiDiscrete([9, 16, 3, 16, 3, 3])`): picks workflow + per-agent tool sets + per-agent token budgets.
- **Prompt policy** (low-level, multi-step, `Discrete(N)`): sequentially appends prompt "atoms" to each agent's system prompt; action `0` = DONE for that agent.

Both policies optimize the same final reward (correctness − efficiency penalties). Supported algorithms are **PPO** (with value head) and **GRPO** (critic-free, group-relative).

## Common commands

Run from repo root (`Context_Optimization/`).

```bash
# Train (API mode — needs OPENROUTER_API_KEY in .env)
python train.py --algorithm ppo  --mask --api --dataset gsm8k --episodes 20000 --api-model qwen/qwen-2.5-7b-instruct
python train.py --algorithm grpo --mask --api --dataset gsm8k --episodes 20000 --entropy-coef 0.08 --tool-bonus 0.15

# Train (local HF model — omit --api)
python train.py --algorithm ppo --dataset gsm8k --episodes 20000 --hf-model Qwen/Qwen2.5-7B-Instruct

# Continue from SFT-pretrained policies (use lower LRs)
python train.py --algorithm ppo --pretrain-structure <path> --pretrain-prompt <path> --struct-lr 1e-4 --prompt-lr 5e-5

# Evaluate trained policies (must match training's API/model setup)
python scripts/eval_hrl.py --structure-model <path> --prompt-model <path> --dataset gsm8k --api --api-model <id> --episodes all --workers 8

# SFT post-training on correct episodes from RL logs
python sft_posttrain.py --rl-log logs/training_log_<algo>_<dataset>_<ts>.json \
  --structure_model_path models/<algo>_models/structure_..._final.pt \
  --prompt_model_path    models/<algo>_models/prompt_..._final.pt \
  --algorithm <ppo|grpo> --epochs 3

# Baselines and ablations
python scripts/baselines/base_model.py --dataset gsm8k --api --api-model <id> --workers 8
python scripts/baselines/dspy/dspy_train.py        # also: dspy_eval.py, gepa_train.py, gepa_eval.py, autogen.py, llm_selector.py
python scripts/exp_2/run_transfer_experiments.py --all --api --api-model <id> --workers 8 --episodes 50
python abalation/embedder_selection/run_all_experiments.py
python abalation/atom_generation/run_all_experiments.py

# One-time setup: precompute MetaCLIP embeddings (avoids GPU overhead during RL)
python scripts/precompute_embeddings.py --all
```

There is **no test runner, linter, or build step**. Smoke-test scripts exist (`agents_system/test_workflows.py`, `tools/test_tools.py`, `utils/test_dataloaders.py`) but are run directly with `python <file>`, not via pytest.

## Environment

- `.env` at repo root: `OPENROUTER_API_KEY` (required for `--api`), optional `OPENROUTER_MODEL`. Loaded explicitly at the top of `train.py` and inside `agents_system/worker.py` — code that touches the API outside these entry points must `load_dotenv` itself.
- `train.py` and `scripts/eval_hrl.py` **hardcode** `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` near the top. Change this if running on a different GPU.
- External binary deps from `requirements.txt`: `tesseract-ocr` (`sudo apt install tesseract-ocr`), playwright chromium (`playwright install --with-deps chromium`, only for the AutoGen baseline).

## Architecture

### Dual-environment, dual-policy flow

`StructureEnv` (`env/structure_env.py`) and `PromptEnv` (`env/prompt_env.py`) are **paired** for one episode:

1. `StructureEnv.step` picks `(workflow_id, agent1_tools, agent1_budget, agent2_tools, agent2_budget, answerer_budget)` and emits an observation but **does not run the LLM**.
2. The structure decision is fed into `PromptEnv`, which then makes one `Discrete` action per prompt slot per agent, advancing through stages `REASONER → VERIFIER → ANSWERER`.
3. Once the prompt policy emits DONE for the answerer (or hits `MAX_PROMPTS_PER_AGENT`, default 3), `PromptEnv` instantiates the chosen workflow and **executes** it end-to-end. The final reward propagates back to **both** policies.

`algorithms/base.BaseTrainer` owns this orchestration; `PPOTrainer` and `GRPOTrainer` subclass it and differ only in update rule (PPO uses a value head + GAE; GRPO uses group-relative advantages, no critic, optional KL to a reference policy refreshed every `--ref-update-every` steps).

### Workflows (`agents_system/workflows/`)

Nine workflows, indexed 0–8, defined twice — once for local HF inference (`hugging_face/`) and once for OpenRouter API calls (`openrouter/`). The active set is selected by `get_workflow` / `get_openrouter_workflow` via `WORKFLOW_REGISTRY` / `OPENROUTER_WORKFLOW_REGISTRY` in `agents_system/workflows/__init__.py`. Workflow 2 (Reason+Verify+Ans) reuses `PromptChainingWorkflow` with `use_verifier=True` — keep this special case in mind when adding workflows.

Workflow IDs (canonical, matches `WORKFLOW_NAMES` in `eval_hrl.py`):
`0 Direct, 1 Reason+Ans, 2 Reason+Verify+Ans, 3 Routing, 4 Parallel-Sectioning, 5 Parallel-Voting, 6 Orchestrator-Workers, 7 Evaluator-Optimizer, 8 Autonomous-Agent`.

### Action masking (`--mask`)

When enabled, the structure policy masks the `agent2_tools` and `agent2_budget` dimensions for workflows that have no second agent (0 Direct, 1 Reason+Ans, 5 Parallel-Voting). Masking logic lives in `StructureEnv._get_action_mask`.

### Tool encoding

The 16-way `agent_tools` action is a 4-bit bitmask: `1=calculator, 2=web_search, 4=python, 8=ocr_reader` (see `tools/registry.py` and `decode_tools` in `scripts/eval_hrl.py`). Workflows invoke tools when the LLM emits `TOOL: <name> || QUERY: <query>` — see `ToolRegistry.parse_and_execute`.

### Prompt atoms (`prompts/library.py`, `prompts/generator.py`)

Each agent role (reasoner / verifier / answerer) has a base set of hand-written atoms keyed by integer (index 0 is always DONE). Per-dataset atoms are auto-generated on first run and cached at `prompts/generated/<dataset>/atoms.json`. Generated atoms are appended to the base dict with new sequential indices (see `_load_from_file`); `NUM_ATOMS` is recomputed via `refresh_counts()`. **Do not rely on a fixed atom count** — `prompt_env` reads `NUM_ATOMS` at runtime.

If atoms are missing on training start, `train.py` spins up a temporary `OpenRouterWorker` using `--prompt-gen-model` (default `openai/gpt-5.2`), falling back to a local Qwen model if the API call fails.

### Workers (`agents_system/worker.py`)

`LLMWorker` (local HF via `transformers`) and `OpenRouterWorker` (HTTP API) implement the same interface (`reason`, `verify`, `answer_direct`, `answer_with_context`, …). Workflows are written against this interface so the same workflow class works in both modes — but the workflow **class itself** is different (`hugging_face/` vs `openrouter/`), selected via the registry. Both workers also expose a `MetaCLIPEmbedder` used for question embeddings; embeddings are read from `embeddings_cache/` if precomputed.

### Datasets (`utils/data_loader/`)

Standard datasets: `gsm8k, hotpotqa, gaia, medqa, aime25, drop`. MMLU is special: any `mmlu_<subject>[_<subject>...]` string is accepted (validated against `MMLU_SUBJECTS`). Use `validate_dataset_name` from `utils.get_dataset` as the argparse `type=` to keep this consistent across scripts.

### Configs (`configs/`)

`load_config(name)` returns a Python module (not a dataclass) with uppercase attributes — `cfg.DATASET_NAME`, `cfg.STRUCTURE_LEARNING_RATE`, etc. Three configs exist (`single_step`, `multi_step`, `hierarchical`) but **only `hierarchical` is actively used**; `single_step` / `multi_step` are baselines for the dual-policy approach and target `GeneralAgentEnv` / `MultiStepAgentEnv` respectively.

### Outputs

- Training: checkpoints under `models/<algo>/<dataset>/<model-slug>/` with `_final` suffix on the last save; logs under `logs/training_log_<algo>_<dataset>_<ts>.json` (consumed by `sft_posttrain.py`).
- Evaluation: `eval_logs/`.
- All of `logs/`, `eval_logs/`, `models/`, `embeddings_cache/`, `paper/`, `gen_prompts/`, `test/` are gitignored.

## Conventions to preserve

- Scripts under `scripts/` insert the repo root into `sys.path` themselves — keep this when adding new scripts there.
- When adding a new workflow, register it in **both** `WORKFLOW_REGISTRY` and `OPENROUTER_WORKFLOW_REGISTRY`, extend `WORKFLOW_NAMES` in `scripts/eval_hrl.py`, and update `StructureEnv.structure_dims[0]` (currently 9) plus any masking logic.
- When adding a new dataset, add a loader in `utils/data_loader/`, register it in `utils/get_dataset.py` (`STANDARD_DATASETS` + dispatch), and rely on first-run auto-generation for the prompt atoms file.
