"""ShapedTau2Env — wraps tau2's AgentGymEnv to add per-turn dense rewards.

Why this exists
---------------
Vanilla tau2 returns reward=0.0 for every mid-episode step and a single
[0,1] scalar at termination. That sparse signal makes RL training slow
and brittle for multi-turn dialogs. This wrapper adds **per-turn shaping
rewards** that fire on every env.step():

    shaped_reward_t = w_action_progress * (newly_matched_gold_actions)
                    + w_tool_error      * (1 if last tool errored else 0)
                    + w_tool_dup        * (1 if same (tool, args) repeats else 0)
                    + w_tokens          * (tokens_used_this_turn)

The terminal reward from tau2 is unchanged; we just add the shaping on
top. With γ-discount in PPO/GRPO, both signals propagate backward to
every earlier policy decision automatically.

Design notes
------------
- Wraps, doesn't fork. Zero changes to upstream tau2.
- Cheap signals only: we never re-run tau2's expensive evaluators
  (DBCheck rebuilds two domain instances per call) mid-episode.
- "action_progress" uses the gold action set from the task — this is
  oracle shaping, FINE for training but DO NOT enable during eval.
  Pass `shaping_cfg={"action_progress": 0.0}` (or use a different env
  class) for eval runs.
- Monotonic by construction: matched gold-action ids never un-match,
  so action_progress can only contribute non-negative deltas.
"""
from __future__ import annotations

import json
from typing import Optional, Set, Tuple

from tau2.gym.gym_agent import AgentGymEnv
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.data_model.message import (
    AssistantMessage,
    ToolMessage,
    UserMessage,
)


_DEFAULT_SHAPING = {
    "action_progress": 0.10,   # +ε per newly matched gold action
    "tool_error":     -0.05,   # −ε if the last tool result errored
    "tool_dup":       -0.02,   # −ε if same (tool, args) emitted twice in a row
    "tokens":         -1e-5,   # per-token penalty
    "terminal_scale":  1.0,    # weight on tau2's terminal reward
}


class ShapedTau2Env(AgentGymEnv):
    """AgentGymEnv with per-turn shaping rewards added via info['shaped_reward'].

    Public additions to step()'s return tuple (vs upstream):
      info["shaped_reward"]      : float, the per-turn shaping computed this step.
      info["matched_action_ids"] : list[str], cumulative set of matched gold actions.
      info["tool_error_step"]    : bool, did the agent's last tool call error.
      info["tool_dup_step"]      : bool, was the agent's last tool call a duplicate.
      info["turn_tokens"]        : int, agent tokens spent this turn.

    The reward returned by step() is:
      mid-episode    : shaped_reward
      terminated step: shaped_reward + terminal_scale * upstream_terminal_reward
    """

    def __init__(
        self,
        *args,
        shaping_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._shaping = {**_DEFAULT_SHAPING, **(shaping_cfg or {})}
        self._matched_action_ids: Set[str] = set()
        self._seen_calls: Set[Tuple[str, str]] = set()
        self._last_token_count: int = 0
        self._task_cache = None

    # --- helpers ---------------------------------------------------------

    def _trajectory(self):
        """Return the running message list as seen by the agent."""
        try:
            obs = self._agent.observation  # property — list[Message]
            return list(obs) if obs else []
        except Exception:
            return []

    def _task(self):
        if self._task_cache is None:
            try:
                self._task_cache = self._get_task()
            except Exception:
                self._task_cache = None
        return self._task_cache

    # --- per-turn shaping deltas -----------------------------------------

    def _delta_action_progress(self) -> Tuple[float, int]:
        """Score newly-matched gold actions since last call. Monotonic."""
        task = self._task()
        if task is None or task.evaluation_criteria is None:
            return 0.0, 0
        gold = task.evaluation_criteria.actions or []
        if not gold:
            return 0.0, 0
        traj = self._trajectory()
        tool_calls = ActionEvaluator.extract_tool_calls(traj)
        gained = 0
        for action in gold:
            if action.action_id in self._matched_action_ids:
                continue
            for tc in tool_calls:
                try:
                    if action.compare_with_tool_call(tc):
                        self._matched_action_ids.add(action.action_id)
                        gained += 1
                        break
                except Exception:
                    # Defensive: never let evaluator quirks crash the env
                    continue
        return self._shaping["action_progress"] * gained, gained

    def _delta_tool_error(self) -> Tuple[float, bool]:
        """−ε if the most recent ToolMessage in the trajectory errored."""
        traj = self._trajectory()
        for m in reversed(traj):
            if isinstance(m, ToolMessage):
                errored = bool(getattr(m, "error", False))
                if errored:
                    return self._shaping["tool_error"], True
                return 0.0, False
        return 0.0, False

    def _delta_tool_dup(self) -> Tuple[float, bool]:
        """−ε if the most-recent agent tool call has been seen before."""
        traj = self._trajectory()
        for m in reversed(traj):
            if isinstance(m, AssistantMessage) and m.is_tool_call():
                last = m.tool_calls[-1]
                try:
                    args_key = json.dumps(last.arguments, sort_keys=True, default=str)
                except Exception:
                    args_key = repr(last.arguments)
                key = (last.name, args_key)
                if key in self._seen_calls:
                    return self._shaping["tool_dup"], True
                self._seen_calls.add(key)
                return 0.0, False
        return 0.0, False

    def _delta_tokens(self) -> Tuple[float, int]:
        """Per-turn token cost: difference in cumulative agent tokens."""
        # Agent doesn't always expose total_tokens; fall back to char count proxy.
        cur = int(getattr(self._agent, "total_tokens", 0) or 0)
        if cur == 0:
            # Approx with last assistant message length / 4 (rough chars→tokens)
            traj = self._trajectory()
            for m in reversed(traj):
                if isinstance(m, AssistantMessage):
                    content = getattr(m, "content", None) or ""
                    cur = self._last_token_count + max(1, len(content) // 4)
                    break
        delta = max(0, cur - self._last_token_count)
        self._last_token_count = cur
        return self._shaping["tokens"] * delta, delta

    # --- gym overrides ---------------------------------------------------

    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self._matched_action_ids.clear()
        self._seen_calls.clear()
        self._last_token_count = 0
        self._task_cache = None
        return out

    def step(self, action: str):
        obs, terminal_reward, terminated, truncated, info = super().step(action)

        d_action, n_matched = self._delta_action_progress()
        d_err,    err_step  = self._delta_tool_error()
        d_dup,    dup_step  = self._delta_tool_dup()
        d_tok,    tok_used  = self._delta_tokens()

        shaping = d_action + d_err + d_dup + d_tok
        info["shaped_reward"]      = shaping
        info["shaped_components"]  = {
            "action_progress": d_action,
            "tool_error":      d_err,
            "tool_dup":        d_dup,
            "tokens":          d_tok,
        }
        info["matched_action_ids"] = sorted(self._matched_action_ids)
        info["matched_action_count_delta"] = n_matched
        info["tool_error_step"]    = err_step
        info["tool_dup_step"]      = dup_step
        info["turn_tokens"]        = tok_used

        if terminated:
            total = shaping + self._shaping["terminal_scale"] * float(terminal_reward or 0.0)
        else:
            total = shaping

        return obs, total, terminated, truncated, info
