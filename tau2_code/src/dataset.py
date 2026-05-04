"""Tau2Dataset - bridges tau2's per-domain task lists into our BaseDataset interface.

Key points:
- get_sample() returns (task_description_text, task_id). The "answer" slot is the task_id;
  the rollout uses it to look up the correct gym task and to cache the final reward.
- evaluate_correctness(prediction, task_id) is a passthrough that reads the cached reward
  set by tau2_dialog_rollout (since reward is computed by tau2's evaluate_simulation,
  not by string-matching the prediction).
- .data exposes a list of dicts so our existing v2 atom generator works without changes
  (it does dict-style .get('question') / .get('answer') lookups).
"""
import random
import sys
import os
from typing import List, Optional

# Ensure project root on path for sibling imports
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.data_loader.base_loader import BaseDataset


def _render_task_description(task) -> str:
    """Build a human-readable single-string description of a tau2 Task for atom grounding
    and as the 'question' our policy/atoms reason over before the dialog starts."""
    parts: List[str] = []
    desc = getattr(task, "description", None)
    if desc is not None:
        if getattr(desc, "purpose", None):
            parts.append(f"Purpose: {desc.purpose}")
        if getattr(desc, "relevant_policies", None):
            parts.append(f"Relevant policies: {desc.relevant_policies}")
        if getattr(desc, "notes", None):
            parts.append(f"Notes: {desc.notes}")
    us = getattr(task, "user_scenario", None)
    if us is not None:
        instr = getattr(us, "instructions", None)
        if instr is not None:
            for fld in ("reason_for_call", "task_instructions", "known_info", "unknown_info"):
                v = getattr(instr, fld, None)
                if v:
                    parts.append(f"{fld}: {v}")
    return "\n".join(parts) if parts else f"Task {task.id}"


def _render_task_answer(task) -> str:
    """Render the evaluation criteria as a string for atom-generation grounding only.
    NOT used at runtime for correctness — that goes through evaluate_simulation."""
    ec = getattr(task, "evaluation_criteria", None)
    if ec is None:
        return ""
    bits = []
    actions = getattr(ec, "actions", None) or []
    if actions:
        names = [getattr(a, "name", "?") for a in actions]
        bits.append(f"Expected actions: {', '.join(names)}")
    ci = getattr(ec, "communicate_info", None) or []
    if ci:
        bits.append(f"Must communicate: {'; '.join(map(str, ci))}")
    nl = getattr(ec, "nl_assertions", None) or []
    if nl:
        bits.append(f"NL assertions: {'; '.join(map(str, nl))}")
    return "\n".join(bits)


class Tau2Dataset(BaseDataset):
    """BaseDataset-shaped view of one tau2 domain split.

    name format: 'tau2_<domain>' (e.g., 'tau2_airline').
    """

    def __init__(self, domain: str, split: str = "train", is_eval: bool = False):
        # Late import so tau2 doesn't need to be installed for non-tau2 datasets.
        from tau2.registry import registry

        if is_eval and split == "train":
            split = "test"
        self.domain = domain
        self.split = split
        self.name = f"tau2_{domain}"

        all_tasks = registry.get_tasks_loader(domain)()
        splits_loader = registry.get_task_splits_loader(domain)
        splits = splits_loader() if splits_loader is not None else {}
        all_ids = [t.id for t in all_tasks]
        keep_ids = set(splits.get(split) or splits.get("base") or all_ids)
        self.tasks = [t for t in all_tasks if t.id in keep_ids]
        if not self.tasks:
            raise ValueError(
                f"Tau2Dataset({domain!r}, split={split!r}) is empty. "
                f"Available split keys: {list(splits.keys())}"
            )

        # dict-shaped view that our v2 atom generator already knows how to read
        self.data = [
            {
                "task_id": t.id,
                "question": _render_task_description(t),
                "answer": _render_task_answer(t),
            }
            for t in self.tasks
        ]

        # Indexed lookup for evaluate_correctness reward retrieval
        self._task_by_id = {t.id: t for t in self.tasks}
        self._reward_by_id: dict = {}

        print(f"[Tau2Dataset] domain={domain} split={split} -> {len(self.tasks)} tasks")

    # ---- BaseDataset interface ----
    def get_sample(self):
        """Returns (task_description_text, task_id). task_id occupies the 'answer' slot
        so the env can pass it back to evaluate_correctness."""
        idx = random.randint(0, len(self.tasks) - 1)
        sample = self.data[idx]
        return sample["question"], sample["task_id"]

    def evaluate_correctness(self, prediction: str, task_id: str) -> float:
        """Returns the reward cached by the most recent rollout for this task_id.
        prediction is ignored — tau2 scores the full simulation, not the final string."""
        return float(self._reward_by_id.get(task_id, 0.0))

    # ---- Tau2 extras (used by the rollout) ----
    def get_task(self, task_id: str):
        """Returns the underlying tau2 Task object."""
        return self._task_by_id[task_id]

    def cache_reward(self, task_id: str, reward: float) -> None:
        """Called by tau2_dialog_rollout after evaluate_simulation runs."""
        self._reward_by_id[task_id] = float(reward)

    def __len__(self):
        return len(self.tasks)
