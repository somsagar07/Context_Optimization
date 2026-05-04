"""Reward helpers for tau2.

We don't recompute reward — tau2's evaluate_simulation already runs at episode end
and is what the paper reports. This module is just a tiny adapter to keep call sites
consistent and to expose the pieces (action match, communicate_info, nl_assertions)
in case we want to log them or use them as shaped intermediate signals later.
"""
from typing import Dict, Any


def normalize_reward(raw_reward: float) -> float:
    """tau2's reward is already in [0, 1]; clamp defensively."""
    if raw_reward is None:
        return 0.0
    try:
        r = float(raw_reward)
    except (TypeError, ValueError):
        return 0.0
    if r < 0.0:
        return 0.0
    if r > 1.0:
        return 1.0
    return r


def extract_reward_breakdown(tau2_info: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the per-component scores out of tau2's evaluation info dict, if present.
    Useful for logging / per-component analysis. Returns empty dict if not available."""
    if not isinstance(tau2_info, dict):
        return {}
    out = {}
    for key in (
        "reward",
        "reward_basis",
        "reward_breakdown",
        "action_checks",
        "communicate_checks",
        "nl_assertions",
        "db_check",
        "env_assertions",
    ):
        if key in tau2_info:
            out[key] = tau2_info[key]
    return out
