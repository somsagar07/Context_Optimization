# ARC: Learning to Configure Agents

Hierarchical reinforcement learning for **learning to configure** multi-agent workflows and prompt selection. The system trains two policies:

- **Structure policy** (high-level): selects workflow type, tool sets, and token budgets.
- **Prompt policy** (low-level): selects prompt atoms per agent in the chosen workflow.

Supported algorithms: **PPO** (with value function) and **GRPO** (critic-free, group-relative).

## Features

- **Dual-policy hierarchy**: structure policy (single-step MultiDiscrete) + prompt policy (multi-step sequential).
- **9 workflows**: Direct, Reason+Ans, Reason+Verify+Ans, Routing, Parallel-Sectioning, Parallel-Voting, Orchestrator-Workers, Evaluator-Optimizer, Autonomous-Agent.
- **Action masking**: optional masking to reduce invalid actions (e.g., mask agent2 for workflows 0, 1, 5).
- **API or local**: train/eval with **OpenRouter API** (e.g. `qwen/qwen-2.5-7b-instruct`) or local Hugging Face models.
- **Multiple benchmarks**: GSM8k, HotPotQA, GAIA, MedQA, AIME25, DROP, and MMLU variants.

## Setup

```bash
cd Context_Optimization
pip install -r requirements.txt
```

**Environment variables** (create a `.env` in the project root or export in shell):

- **API mode (OpenRouter)**  
  - `OPENROUTER_API_KEY` — required when using `--api` and `--api-model`.  
  - Optional: `OPENROUTER_MODEL` for default model.

- **Optional**: `HF_HOME` or `TRANSFORMERS_CACHE` for Hugging Face model cache.

Example `.env`:

```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o
```

## Datasets

Supported `--dataset` values:

- **Standard**: `gsm8k`, `hotpotqa`, `gaia`, `medqa`, `aime25`, `drop`
- **MMLU**: `mmlu`, or any `mmlu_<subject>` combination (e.g. `mmlu_math`, `mmlu_physics`)

Prompt atoms are stored under `prompts/generated/<dataset>/atoms.json` and are loaded or generated at first run.

## Training

From the project root (`Context_Optimization/`):

```bash
# PPO with action masking and OpenRouter API (e.g. Qwen 7B)
python train.py --algorithm ppo --mask --api --dataset gsm8k --episodes 10 \
  --entropy-coef 0.3 --tool-bonus 0.2 --batch-size 64 \
  --api-model qwen/qwen-2.5-7b-instruct

# GRPO (no value head)
python train.py --algorithm grpo --mask --api --dataset gsm8k --episodes 20000 \
  --entropy-coef 0.08 --tool-bonus 0.15 --api-model qwen/qwen-2.5-7b-instruct

# Local Hugging Face model (no --api)
python train.py --algorithm ppo --dataset gsm8k --episodes 20000 --hf-model Qwen/Qwen2.5-7B-Instruct

# From pretrained (e.g. SFT) with lower LRs
python train.py --algorithm ppo --dataset gsm8k --episodes 20000 \
  --pretrain-structure models/sft_posttrained/structure_policy_sft.pt \
  --pretrain-prompt models/sft_posttrained/prompt_policy_sft.pt \
  --struct-lr 1e-4 --prompt-lr 5e-5
```

Important flags:

- `--algorithm`: `ppo` or `grpo`
- `--mask`: enable action masking
- `--api`: use OpenRouter; requires `OPENROUTER_API_KEY`
- `--api-model`: OpenRouter model ID (e.g. `qwen/qwen-2.5-7b-instruct`, `openai/gpt-4o`)
- `--dataset`: benchmark name (default from config)
- `--episodes`, `--batch-size`, `--entropy-coef`, `--tool-bonus`: training hyperparameters
- `--num-workers`: parallel workers for API mode only

Checkpoints and logs are written under `models/` and `logs/`. The script prints an evaluation command when training finishes.

## Evaluation

Evaluate trained structure and prompt policies:

```bash
# Match training: same dataset and API/model as training
python scripts/eval_hrl.py \
  --structure-model models/ppo/gsm8k/qwen-2_5-7b-instruct/structure_policy_gsm8k_<timestamp>_final.pt \
  --prompt-model models/ppo/gsm8k/qwen-2_5-7b-instruct/prompt_policy_gsm8k_<timestamp>_final.pt \
  --dataset gsm8k --api --api-model qwen/qwen-2.5-7b-instruct \
  --episodes 10 --workers 2
```

- **Hugging Face (no API)**  
  Omit `--api` and `--api-model`; use the same model as in training.

- **Full test set**  
  Use `--episodes all` to evaluate on all test samples.

- **Parallel workers**  
  `--workers N` speeds up API-mode evaluation (e.g. `--workers 12`). Use a moderate value (e.g. 2–8) to avoid rate limits.

Results and workflow counts are printed to the console; logs are saved under `eval_logs/`.

## Project structure

| Path | Description |
|------|-------------|
| `train.py` | Unified training entry (PPO/GRPO, API/local) |
| `configs/` | Base, hierarchical, single-step, multi-step configs |
| `algorithms/` | PPO and GRPO trainers; base RL logic |
| `env/` | Structure env (high-level) and prompt env (low-level) |
| `agents_system/` | Workflows (Direct, Routing, Autonomous-Agent, etc.) and workers (Hugging Face, OpenRouter) |
| `prompts/` | Prompt atoms and generation; per-dataset `generated/<dataset>/atoms.json` |
| `utils/data_loader/` | Dataset loaders (GSM8k, HotPotQA, GAIA, MedQA, AIME25, DROP, MMLU) |
| `scripts/eval_hrl.py` | Hierarchical policy evaluation |
| `ReadMe_Instructions/` | ACTION_MASKING, WORKFLOWS, SFT_POSTTRAIN docs |

