"""Multi-turn dialog rollout for tau2 tasks.

Drives the loop: agent.respond(history) -> env.step(action) -> repeat.
Translates our internal action format ('TOOL: name || QUERY: <json>') into tau2's
expected gym-action format (functional notation like name(arg='val')). Plain messages
are passed through unchanged.

Returns (transcript_text, exec_info, reward) shaped to match what PromptEnv expects.
exec_info also carries the per-component breakdown extracted from
info['reward_info'] so the trainer can compute a shaped reward instead of just
binarizing the scalar.
"""
import json
import os
import re
from typing import Dict, Tuple


def _safe_int_count(items, predicate=lambda x: True) -> int:
    if not items:
        return 0
    n = 0
    for it in items:
        try:
            if predicate(it):
                n += 1
        except Exception:
            pass
    return n


def _parse_reward_info(info: Dict) -> Dict:
    """Extract per-component pass/fail counts from tau2's RewardInfo JSON blob.

    info['reward_info'] is set by AgentGymEnv at episode end (gym_agent.py:821).
    It's a JSON string of a pydantic model with action_checks, communicate_checks,
    db_check, env_assertions, nl_assertions, reward_basis.

    Returns a flat dict with integer counts and a list of reward_basis strings,
    plus the raw original reward and termination reason for logging.
    """
    out = {
        "tau2_action_pass": 0, "tau2_action_total": 0, "tau2_action_missed": 0,
        "tau2_communicate_pass": 0, "tau2_communicate_total": 0, "tau2_communicate_missed": 0,
        "tau2_env_pass": 0, "tau2_env_total": 0, "tau2_env_missed": 0,  # merged DB + env_assertions
        "tau2_nl_pass": 0, "tau2_nl_total": 0, "tau2_nl_missed": 0,
        "tau2_reward_basis": [],
        "tau2_original_reward": 0.0,
        "tau2_termination_reason": "",
    }
    if not isinstance(info, dict):
        return out
    raw = info.get("reward_info")
    if not raw:
        return out
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return out
    if not isinstance(parsed, dict):
        return out

    out["tau2_original_reward"] = float(parsed.get("reward") or 0.0)
    rb = parsed.get("reward_basis") or []
    out["tau2_reward_basis"] = [str(x) for x in rb] if isinstance(rb, list) else []

    # NB: tau2 uses different boolean field names per check type. ActionCheck has
    # 'action_match', DBCheck has 'db_match', everything else has 'met'.

    # Action checks
    actions = parsed.get("action_checks") or []
    out["tau2_action_total"] = len(actions)
    out["tau2_action_pass"] = _safe_int_count(actions, lambda a: bool(a.get("action_match")))
    out["tau2_action_missed"] = out["tau2_action_total"] - out["tau2_action_pass"]

    # Communicate checks
    comms = parsed.get("communicate_checks") or []
    out["tau2_communicate_total"] = len(comms)
    out["tau2_communicate_pass"] = _safe_int_count(comms, lambda c: bool(c.get("met")))
    out["tau2_communicate_missed"] = out["tau2_communicate_total"] - out["tau2_communicate_pass"]

    # NL assertions
    nls = parsed.get("nl_assertions") or []
    out["tau2_nl_total"] = len(nls)
    out["tau2_nl_pass"] = _safe_int_count(nls, lambda n: bool(n.get("met")))
    out["tau2_nl_missed"] = out["tau2_nl_total"] - out["tau2_nl_pass"]

    # Merge DB + env_assertions into a single "env" component (matches paper grouping)
    db = parsed.get("db_check")
    env_assertions = parsed.get("env_assertions") or []
    env_total = (1 if db is not None else 0) + len(env_assertions)
    env_pass = 0
    if db is not None and bool(db.get("db_match")):
        env_pass += 1
    env_pass += _safe_int_count(env_assertions, lambda e: bool(e.get("met")))
    out["tau2_env_total"] = env_total
    out["tau2_env_pass"] = env_pass
    out["tau2_env_missed"] = env_total - env_pass

    # Termination reason (from the simulation, surfaced via reward_info.info.note for premature)
    sub_info = parsed.get("info") or {}
    if isinstance(sub_info, dict):
        note = sub_info.get("note") or ""
        if "Termination reason:" in str(note):
            out["tau2_termination_reason"] = str(note).split("Termination reason:")[-1].strip().rstrip(".")
    return out


_TOOL_LINE_RE = re.compile(
    r"TOOL:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\|\|\s*QUERY:\s*([^\n]+)"
)


def _action_to_tau2_format(action_text: str) -> str:
    """Translate our 'TOOL: name || QUERY: {...}' format into tau2's preferred format.

    Tau2 accepts JSON ({"name":..., "arguments":...}), so we emit JSON for tool calls
    (most robust). Plain messages are passed through untouched.
    """
    if not action_text:
        return ""
    m = _TOOL_LINE_RE.search(action_text)
    if not m:
        return action_text  # plain message
    tool_name = m.group(1).strip()
    raw_args = m.group(2).strip()

    # Try to parse JSON arguments; if it fails, send the raw text as a single 'query' arg.
    try:
        args = json.loads(raw_args)
        if not isinstance(args, dict):
            args = {"value": args}
    except json.JSONDecodeError:
        args = {"query": raw_args}

    return json.dumps({"name": tool_name, "arguments": args})


def _format_observation(obs) -> str:
    """tau2 returns either a string or a list of message objects depending on the
    `all_messages_as_observation` flag. Coerce to a string."""
    if isinstance(obs, str):
        return obs
    try:
        return "\n".join(str(m) for m in obs)
    except Exception:
        return str(obs)


def tau2_dialog_rollout(
    agent,
    domain: str,
    task_id: str,
    *,
    max_turns: int = 12,
    solo_mode: bool = False,
    user_llm: str = None,
    total_token_cap: int = 200_000,
    use_shaped_rewards: bool = True,
    shaping_mode: str = "training",
    shaping_cfg: Dict = None,
) -> Tuple[str, Dict, float]:
    """Run a single tau2 episode.

    Args:
        agent: ConfiguredAgent. Must expose .respond(history) -> str and .stats() -> dict.
        domain: tau2 domain name (e.g. 'airline').
        task_id: tau2 task id (string).
        max_turns: hard cap on agent turns (default 12 ~ 6 round-trips).
        solo_mode: if True, no user simulator (agent acts autonomously).
        user_llm: LiteLLM model id for the user simulator (e.g. 'openrouter/openai/gpt-4o-mini').
                  Falls back to env var OPENROUTER_USER_MODEL.
        total_token_cap: hard cap on cumulative agent token usage; episode ends if exceeded.
        use_shaped_rewards: if True (default for tau2_*), use ShapedTau2Env to add per-turn
                            dense shaping rewards (action_progress, tool_error, tool_dup, tokens).
                            If False, use stock AgentGymEnv (paper-faithful, sparse).
        shaping_mode: "training" (full shaping incl. oracle action_progress),
                      "eval" (zero out oracle signals; raw tau2 reward only — paper-faithful),
                      "off" (alias for use_shaped_rewards=False).
        shaping_cfg: optional dict of per-component weight overrides. Merged on top of
                     mode defaults.

    Returns:
        (transcript, exec_info, reward) where reward is the scalar episode reward.
        When use_shaped_rewards is on, this scalar is (sum_of_per_turn_shaping +
        tau2_terminal_reward) — single-config-per-episode architecture absorbs all
        per-turn shaping into the terminal signal. Per-turn detail is logged in
        exec_info["per_turn_shaping"] for transparency.
    """
    import gymnasium as gym
    from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID

    register_gym_agent()

    user_llm = user_llm or os.environ.get("OPENROUTER_USER_MODEL", "openrouter/openai/gpt-4o-mini")

    if shaping_mode == "off":
        use_shaped_rewards = False

    env_kwargs = dict(
        domain=domain,
        task_id=task_id,
        solo_mode=solo_mode,
        max_steps=max(2 * max_turns, 8),  # tau2's max_steps counts both sides
    )
    if not solo_mode:
        env_kwargs["user_llm"] = user_llm

    if use_shaped_rewards:
        # Build effective shaping config based on mode.
        # eval mode: zero out oracle signals (action_progress uses gold action set,
        # which is leakage at eval time). Keep terminal_scale=1.0 so total reward
        # matches paper. Other components (tool_error/dup/tokens) are local and
        # safe to keep, though by default we zero them too in eval for strict
        # paper-faithfulness; pass shaping_cfg to override.
        if shaping_mode == "eval":
            mode_defaults = {
                "action_progress": 0.0,
                "tool_error": 0.0,
                "tool_dup": 0.0,
                "tokens": 0.0,
                "terminal_scale": 1.0,
            }
        else:  # "training"
            mode_defaults = {}  # uses ShapedTau2Env._DEFAULT_SHAPING
        merged_cfg = dict(mode_defaults)
        if shaping_cfg:
            merged_cfg.update(shaping_cfg)
        from tau2_code.src.shaped_env import ShapedTau2Env
        env = ShapedTau2Env(**env_kwargs, shaping_cfg=merged_cfg)
    else:
        env = gym.make(TAU_BENCH_ENV_ID, **env_kwargs)

    last_reward = 0.0
    last_info = {}
    transcript_parts = []
    # Tau2 mode: ConfiguredAgent passes empty tool lists to the inner workflow
    # so workflow._process_tool_calls short-circuits (tau2's gym env is the
    # executor). That makes agent.stats()["tools_count"] always 0 — useless for
    # tau2 logging. Count tool-call ATTEMPTS at the rollout layer instead by
    # scanning the agent's per-turn output for our `TOOL: ... || QUERY: ...`
    # format. This is the count we surface as exec_info["tools_count"].
    tool_call_attempts = 0
    # Per-turn shaping log when ShapedTau2Env is active (empty list otherwise)
    per_turn_shaping = []
    per_turn_components = []

    try:
        obs, info = env.reset()
        transcript_parts.append(_format_observation(obs))

        for turn in range(max_turns):
            history = "\n".join(transcript_parts).strip()
            action_internal = agent.respond(history)

            # Total-token guardrail
            if agent.cumulative_tokens > total_token_cap:
                transcript_parts.append("[system] Token budget exhausted; ending dialog.")
                break

            if _TOOL_LINE_RE.search(action_internal or ""):
                tool_call_attempts += 1

            action_tau2 = _action_to_tau2_format(action_internal)
            transcript_parts.append(f"agent: {action_internal}")

            obs, reward, terminated, truncated, info = env.step(action_tau2)
            last_reward = float(reward) if reward is not None else 0.0
            last_info = info or {}

            # Track per-turn shaping when ShapedTau2Env is active. Stock
            # AgentGymEnv puts no "shaped_reward" in info, so list stays empty.
            if isinstance(info, dict) and "shaped_reward" in info:
                per_turn_shaping.append(float(info.get("shaped_reward", 0.0)))
                per_turn_components.append(dict(info.get("shaped_components", {})))

            obs_text = _format_observation(obs)
            if obs_text:
                transcript_parts.append(obs_text)

            if terminated or truncated:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    transcript = "\n".join(transcript_parts).strip()
    stats = agent.stats()
    # For tau2, prefer the rollout-level attempt count over the workflow-level
    # zero. For non-tau2 callers (none today, but kept for safety) fall back to
    # the agent's own counter.
    tools_count = tool_call_attempts if tool_call_attempts > 0 else int(stats.get("tools_count", 0))
    # Sum of per-turn shaping (already baked into last_reward at the
    # terminated step by ShapedTau2Env.step). We re-export it as a
    # standalone scalar so the trainer's logger and downstream consumers
    # can attribute reward to shaping vs terminal.
    total_shaping = float(sum(per_turn_shaping)) if per_turn_shaping else 0.0
    exec_info = {
        "steps": max(stats.get("steps", 0), 1),
        "tools_count": tools_count,
        "tools_attempted": tool_call_attempts,
        "total_tokens": stats.get("total_tokens", 0),
        "valid_code_count": 0,
        "file_access_count": 0,
        "tau2_reward": last_reward,
        # NOTE: deliberately NOT exporting last_info — it contains tau2's Task
        # pydantic model which isn't JSON-serializable. _parse_reward_info()
        # below pulls everything we need (per-component pass/total/missed +
        # termination reason) into plain types.
        "tau2_turns": turn + 1 if "turn" in dir() else 0,
        # Shaping diagnostics. Prefixed tau2_ so they propagate through the
        # tau2_*-only forwarding filter in prompt_env._step.
        "tau2_per_turn_shaping": per_turn_shaping,
        "tau2_per_turn_components": per_turn_components,
        "tau2_total_shaping": total_shaping,
        "tau2_shaping_mode": shaping_mode if use_shaped_rewards else "off",
    }
    # Per-component breakdown for shaped reward + logging
    exec_info.update(_parse_reward_info(last_info))
    return transcript, exec_info, last_reward
