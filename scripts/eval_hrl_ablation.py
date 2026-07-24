"""Per-decision-type ablation for ARC (rebuttal E3).

Evaluates a trained hierarchical policy while FREEZING one decision dimension
to a static default, isolating each decision type's contribution:

  --freeze-workflow K   structure action[0] = K for every query
  --freeze-tools K      structure action[1] and [3] (agent tool bitmasks) = K
  --freeze-budget K     structure action[2], [4], [5] (budget tiers) = K
  --freeze-prompt       prompt policy always emits DONE (no learned atoms)

Everything else (checkpoint loading, atom handling, dataset iteration, logging)
is inherited verbatim from scripts/eval_hrl.py by wrapping its policy loaders
and calling its main(). All eval_hrl.py flags are accepted and passed through.

Example (leave-one-out, one dimension per run):
  python scripts/eval_hrl_ablation.py --structure-model M1 --prompt-model M2 \
      --dataset gsm8k --api --api-model qwen/qwen-2.5-7b-instruct \
      --episodes all --workers 8 --freeze-workflow 2
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _extract_flag(argv, name, has_value):
    """Remove --name [value] from argv; return value (or True/None)."""
    if name not in argv:
        return argv, None
    i = argv.index(name)
    if has_value:
        val = argv[i + 1]
        return argv[:i] + argv[i + 2:], val
    return argv[:i] + argv[i + 1:], True


class FrozenStructurePolicy:
    """Wraps a structure policy; overrides selected MultiDiscrete dims.

    Structure action layout (see StructureEnv / repo CLAUDE.md):
      [0] workflow_id (9), [1] agent1_tools (16), [2] agent1_budget (3),
      [3] agent2_tools (16), [4] agent2_budget (3), [5] answerer_budget (3)
    """

    def __init__(self, base, overrides):
        self._base = base
        self._overrides = overrides  # dict: action_index -> frozen value

    def get_action(self, obs, deterministic=True, temperature=1.0):
        action = self._base.get_action(obs, deterministic, temperature)
        action = np.array(action, dtype=np.int64).copy()
        for idx, val in self._overrides.items():
            action[idx] = val
        return action

    def __getattr__(self, name):
        return getattr(self._base, name)


class DonePromptPolicy:
    """Always emits action 0 (DONE): no prompt atoms are ever appended."""

    def __init__(self, base):
        self._base = base

    def get_action(self, obs, deterministic=True, temperature=1.0):
        return 0

    def __getattr__(self, name):
        return getattr(self._base, name)


def main():
    argv = sys.argv[1:]
    argv, fw = _extract_flag(argv, "--freeze-workflow", True)
    argv, ft = _extract_flag(argv, "--freeze-tools", True)
    argv, fb = _extract_flag(argv, "--freeze-budget", True)
    argv, fp = _extract_flag(argv, "--freeze-prompt", False)

    overrides = {}
    if fw is not None:
        overrides[0] = int(fw)
    if ft is not None:
        overrides[1] = int(ft)
        overrides[3] = int(ft)
    if fb is not None:
        overrides[2] = int(fb)
        overrides[4] = int(fb)
        overrides[5] = int(fb)

    frozen_desc = []
    if fw is not None: frozen_desc.append(f"workflow={fw}")
    if ft is not None: frozen_desc.append(f"tools={ft}")
    if fb is not None: frozen_desc.append(f"budget={fb}")
    if fp: frozen_desc.append("prompt=DONE")
    print(f"[ablation] frozen dims: {', '.join(frozen_desc) if frozen_desc else 'none (full policy)'}")

    sys.argv = [sys.argv[0]] + argv

    from scripts import eval_hrl

    orig_load_structure = eval_hrl.load_structure_policy
    orig_load_prompt = eval_hrl.load_prompt_policy

    def load_structure_wrapped(path, device="cpu"):
        policy, algo = orig_load_structure(path, device)
        if overrides:
            policy = FrozenStructurePolicy(policy, overrides)
        return policy, algo

    def load_prompt_wrapped(path, device="cpu"):
        policy, algo = orig_load_prompt(path, device)
        if fp:
            policy = DonePromptPolicy(policy)
        return policy, algo

    eval_hrl.load_structure_policy = load_structure_wrapped
    eval_hrl.load_prompt_policy = load_prompt_wrapped
    eval_hrl.main()


if __name__ == "__main__":
    main()
