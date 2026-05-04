# tau2_code

Tau2-bench integration for the Context_Optimization hierarchical RL framework.

## Layout

```
tau2_code/
  _tau2_upstream/      # vendored clone of sierra-research/tau2-bench (editable install)
  src/                 # our integration layer
    dataset.py             Tau2Dataset       BaseDataset over a domain split
    tool_registry.py       Tau2ToolRegistry  ToolRegistry-shaped facade over tau2 tools
    tool_groups.py         Semantic tool groupings per domain (bitmask over groups)
    configured_agent.py    ConfiguredAgent   Wraps any of our 9 workflows + atoms + tools
    rollout.py             tau2_dialog_rollout  Multi-turn dialog driver, returns reward
    shaped_env.py          ShapedTau2Env     Per-turn dense reward shaping wrapper
    reward.py              Helpers around evaluate_simulation
  scripts/
    smoke_test_mock.py     End-to-end sanity check on the mock domain
    eval_tau2.py           Full eval driver (base, hrl agents; multi-trial pass^k)
```

## Install (one-time)

From the repo root, with your conda env active:

```bash
pip install -e "tau2_code/_tau2_upstream[gym]"
```

If the upstream's `pyproject.toml` requires a newer Python than your env, edit
`tau2_code/_tau2_upstream/pyproject.toml`'s `requires-python` line — only the
typing-only feature `@override` is used and that's already imported from
`typing_extensions`, so 3.11+ is fine.

## .env additions

```
OPENROUTER_API_KEY=...                            # already required for our pipeline
OPENROUTER_USER_MODEL=openrouter/openai/gpt-4o-mini   # user-simulator LLM for tau2
```

## Smoke test

```bash
python tau2_code/scripts/smoke_test_mock.py
```

Should print one mock-domain transcript and a final reward in [0, 1].

## How it slots into the existing trainer

When `cfg.DATASET_NAME` starts with `tau2_`:

- `utils/get_dataset.py` dispatches to `Tau2Dataset(domain=...)`.
- `tools.get_tool_registry(name)` returns `Tau2ToolRegistry(domain)` instead of the
  default 4-tool registry. Structure policy picks a bitmask over semantic tool groups
  (2^G actions with G ~= 4 per domain).
- `prompts/library.py` reads atoms from `prompts/generated_v2/tau2_<domain>/atoms.json`
  (same convention as standard datasets).
- `env/structure_env.py` sizes the tool action dimension to 2^G for the domain's G
  semantic groups (instead of the fixed 16 for default datasets).
- `env/prompt_env.py._execute_workflow` branches on the dataset prefix and calls
  `tau2_dialog_rollout(...)` instead of the single-shot workflow.execute(). The reward
  comes back through `Tau2Dataset.cache_reward()` -> `evaluate_correctness()`.
- `algorithms/base.py` has a tau2-specific reward path in `_compute_episode_reward()`
  that replaces the binary-correctness formula with a multi-stage shaped reward
  (weighted component fractions + completion bonus + per-pass/miss signals).
- `train.py` adds tau2-specific CLI args (`--tau2-max-turns`, reward weights,
  `--per-turn-config` which is ON by default for tau2 datasets).
- `scripts/eval_hrl.py` adds obs_dim-aware atom loading for backward-compat with
  checkpoints trained before the NUM_ATOMS fix.

### Per-turn reconfiguration (default for tau2)

By default, tau2 training uses per-turn reconfiguration: the structure + prompt
policies are re-run at every dialog turn (not just once per episode). This lets
the policy adapt its workflow/tools/atoms as the conversation evolves. Each turn
produces (state, action, reward) tuples for the PPO/GRPO buffer.

Disable with `--no-per-turn-config` to use single-config mode (pick once at start).

## Eval

```bash
# Base model baseline
python tau2_code/scripts/eval_tau2.py --domain airline --split test \
    --agent-type base --agent-llm qwen/qwen-2.5-7b-instruct \
    --episodes 10 --workers 10 --num-trials 1

# Trained HRL policy
python tau2_code/scripts/eval_tau2.py --domain airline --agent-type hrl \
    --structure-model <path> --prompt-model <path> \
    --api-model qwen/qwen-2.5-7b-instruct --workers 8

# Multi-trial for pass^k (paper metric)
python tau2_code/scripts/eval_tau2.py --domain airline --agent-type base \
    --agent-llm qwen/qwen-2.5-7b-instruct --episodes all --workers 8 --num-trials 4
```

Eval uses `shaping_mode="eval"` (zeros oracle signals). Both the paper-faithful
tau2 reward (`tau2_reward`) and the shaped reward (`tau2_reward_shaped`) are logged
per episode. `binary_pass` is computed from the paper reward only.
