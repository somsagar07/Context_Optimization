"""Per-turn workflow reconfiguration for tau2 dialogs.

Standard rollout: pick (workflow, tools, budgets, prompts) once per episode.
The dialog runs to completion, then a single terminal reward arrives.

Per-turn rollout (this module): pick those things FRESH at every dialog turn,
adapting to the evolving conversation. Each turn produces:
  - 1 structure-policy decision (workflow + tools + budgets)
  - up to 3*MAX_PROMPTS_PER_AGENT prompt-policy decisions (atom + DONE per role)
  - 1 workflow execution → 1 agent action
  - 1 env.step → 1 shaped reward (from ShapedTau2Env) + maybe terminal reward

The function returns a trajectory dict shaped to feed into the existing
PPO/GRPO update — each turn contributes (s,a,r) tuples to the buffer.

Why a function (not a gym Env): the per-turn loop alternates between policy
inference, env stepping, and re-embedding state. That's awkward to fit into
the gym Env contract (which assumes the env owns the action space and the
caller owns the policy). A function is the natural fit.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


_TOOL_LINE_RE = re.compile(
    r"TOOL:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\|\|\s*QUERY:\s*([^\n]+)"
)


def _action_to_tau2_format(action_text: str) -> str:
    """Translate our 'TOOL: name || QUERY: {...}' format into tau2's preferred format."""
    if not action_text:
        return ""
    m = _TOOL_LINE_RE.search(action_text)
    if not m:
        return action_text
    tool_name = m.group(1).strip()
    raw_args = m.group(2).strip()
    try:
        args = json.loads(raw_args)
        if not isinstance(args, dict):
            args = {"value": args}
    except json.JSONDecodeError:
        args = {"query": raw_args}
    return json.dumps({"name": tool_name, "arguments": args})


def _format_observation(obs) -> str:
    if isinstance(obs, str):
        return obs
    try:
        return "\n".join(str(m) for m in obs)
    except Exception:
        return str(obs)


_EMBED_WARNING_SHOWN = False


def _embed_state(worker, transcript_text: str, task_brief: str, fallback_emb=None) -> np.ndarray:
    """Compute a state embedding from the running transcript.

    Falls back to the task-brief embedding when transcript is empty (turn 0).
    Errors fall back to fallback_emb if provided, otherwise the task brief
    embedded fresh.
    """
    global _EMBED_WARNING_SHOWN
    text = transcript_text.strip()
    if not text:
        text = task_brief
    try:
        return worker.get_embedding(text[-2000:])
    except Exception as e:
        if not _EMBED_WARNING_SHOWN:
            print(f"[multiturn_rollout] WARNING: embedding failed, using fallback. "
                  f"Per-turn state will be STATIC. Error: {e}")
            _EMBED_WARNING_SHOWN = True
        if fallback_emb is not None:
            return fallback_emb
        return worker.get_embedding(task_brief[:1000])


def multiturn_rollout(
    *,
    structure_policy,
    prompt_policy,
    structure_env,
    prompt_env,
    domain: str,
    task_id: str,
    deterministic: bool = False,
    max_turns: int = 8,
    solo_mode: bool = False,
    user_llm: Optional[str] = None,
    use_shaped_rewards: bool = True,
    shaping_mode: str = "training",
    shaping_cfg: Optional[dict] = None,
    total_token_cap: int = 200_000,
    use_action_masking: bool = False,
) -> Dict:
    """Drive a per-turn-reconfigured tau2 dialog.

    Args:
        structure_policy: object exposing .get_action(obs, deterministic=False) ->
                          (action_array, log_prob, value, action_mask)
        prompt_policy:    object exposing .get_action(obs, deterministic=False) ->
                          (action, log_prob, value)
        structure_env:    StructureEnv instance, used for action-mask computation
                          and obs construction.
        prompt_env:       PromptEnv instance with skip_execute=True semantics.
        domain, task_id: tau2 domain + task id.
        deterministic:    pass-through to policies.
        max_turns:        hard cap on dialog turns.
        solo_mode:        tau2 solo mode flag.
        use_shaped_rewards: wrap env with ShapedTau2Env (training default ON).
        shaping_mode:     "training" | "eval" | "off".

    Returns:
        Dict with these keys (per-turn lists indexed by turn):
            structure_obs_list, structure_actions, structure_log_probs,
            structure_values, structure_action_masks
            prompt_obs_list (flat across turns), prompt_actions, prompt_log_probs,
            prompt_values
            per_turn_rewards: list[float]      one reward per dialog turn
            per_turn_components: list[dict]    shaping breakdown per turn
            terminal_reward: float             tau2's terminal reward
            total_reward: float                sum(per_turn_rewards) + terminal
            transcript: str
            exec_info: dict                    diagnostics (tau2_*, num_turns, etc.)
    """
    # Local imports to avoid hard dependency at module-load time.
    from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID
    register_gym_agent()

    # Eagerly load MetaCLIP to GPU — per-turn transcripts won't hit the
    # precomputed cache, so the model MUST be ready for on-the-fly embedding.
    if hasattr(prompt_env, "worker") and hasattr(prompt_env.worker, "embedder"):
        prompt_env.worker.embedder.ensure_loaded()

    user_llm = user_llm or os.environ.get(
        "OPENROUTER_USER_MODEL", "openrouter/openai/gpt-4o-mini"
    )

    if shaping_mode == "off":
        use_shaped_rewards = False

    env_kwargs = dict(
        domain=domain, task_id=task_id, solo_mode=solo_mode,
        max_steps=max(2 * max_turns, 8),
    )
    if not solo_mode:
        env_kwargs["user_llm"] = user_llm

    if use_shaped_rewards:
        from tau2_code.src.shaped_env import ShapedTau2Env
        if shaping_mode == "eval":
            mode_defaults = {
                "action_progress": 0.0, "tool_error": 0.0, "tool_dup": 0.0,
                "tokens": 0.0, "terminal_scale": 1.0,
            }
        else:
            mode_defaults = {}
        merged_cfg = dict(mode_defaults)
        if shaping_cfg:
            merged_cfg.update(shaping_cfg)
        env = ShapedTau2Env(**env_kwargs, shaping_cfg=merged_cfg)
    else:
        import gymnasium as gym
        env = gym.make(TAU_BENCH_ENV_ID, **env_kwargs)

    # Trajectory accumulators
    structure_obs_list, structure_actions, structure_log_probs = [], [], []
    structure_values, structure_action_masks = [], []
    prompt_obs_list, prompt_actions, prompt_log_probs, prompt_values = [], [], [], []
    per_turn_rewards: List[float] = []
    per_turn_components: List[dict] = []

    transcript_parts: List[str] = []
    last_terminal_reward = 0.0
    last_info: dict = {}
    tool_call_attempts = 0

    # Imported here for atom-text resolution (matches PromptEnv._execute_workflow path)
    from prompts import library
    from tau2_code.src import ConfiguredAgent

    try:
        obs, info = env.reset()
        transcript_parts.append(_format_observation(obs))
        task_brief = transcript_parts[0]
        # Cache an initial task-brief embedding for fallback
        try:
            initial_emb = prompt_env.worker.get_embedding(task_brief[:1000])
        except Exception:
            initial_emb = np.zeros(
                prompt_env.worker.model.config.hidden_size, dtype=np.float32
            )

        for turn in range(max_turns):
            # 1. Build state embedding from current transcript
            transcript_text = "\n".join(transcript_parts).strip()
            state_emb = _embed_state(
                prompt_env.worker, transcript_text, task_brief, fallback_emb=initial_emb
            )

            # 2. Structure decision (high-level policy)
            #    Reuse StructureEnv to build the obs + action mask.
            structure_env.question_embedding = state_emb
            structure_env.current_q = task_brief  # for any downstream reads
            struct_obs = structure_env._get_observation()
            action_mask = structure_env._get_action_mask() if use_action_masking else None
            if action_mask is not None:
                s_out = structure_policy.get_action(struct_obs, deterministic, action_mask=action_mask)
            else:
                s_out = structure_policy.get_action(struct_obs, deterministic)
            # Training policies return (action, log_prob, value); eval policies return just the action.
            if isinstance(s_out, tuple):
                struct_action, struct_log_prob, struct_value = s_out
            else:
                struct_action, struct_log_prob, struct_value = s_out, 0.0, 0.0
            workflow_depth = int(struct_action[0])
            agent1_tools_idx = int(struct_action[1])
            agent1_budget_idx = int(struct_action[2])
            agent2_tools_idx = int(struct_action[3])
            agent2_budget_idx = int(struct_action[4])
            answerer_budget_idx = int(struct_action[5])

            structure_obs_list.append(np.asarray(struct_obs, dtype=np.float32).copy())
            structure_actions.append(np.asarray(struct_action))
            structure_log_probs.append(float(struct_log_prob))
            structure_values.append(float(struct_value))
            structure_action_masks.append(action_mask)

            # 3. Prompt decisions (low-level multi-step policy).
            #    Configure prompt_env with the structure decision + this turn's
            #    state embedding, then run its multi-step atom-selection loop
            #    with skip_execute=True so it terminates without running the
            #    workflow.
            prompt_env.current_q = task_brief
            prompt_env.current_a = task_id
            prompt_env.question_embedding = state_emb
            prompt_env.workflow_depth = workflow_depth
            prompt_env.agent1_tools_idx = agent1_tools_idx
            prompt_env.agent1_budget_idx = agent1_budget_idx
            prompt_env.agent2_tools_idx = agent2_tools_idx
            prompt_env.agent2_budget_idx = agent2_budget_idx
            prompt_env.answerer_budget_idx = answerer_budget_idx
            if workflow_depth in (0, 5):
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_ANSWERER
            else:
                prompt_env.prompt_stage = prompt_env.PROMPT_STAGE_REASONER
            prompt_env.prompt_step = 0
            prompt_env.selected_prompts = {"reasoner": [], "verifier": [], "answerer": []}
            prompt_env._structure_set = True
            prompt_env.skip_execute = True
            try:
                prompt_obs = prompt_env._get_observation()
                done = False
                while not done:
                    p_out = prompt_policy.get_action(prompt_obs, deterministic)
                    if isinstance(p_out, tuple):
                        p_action, p_log_prob, p_value = p_out
                    else:
                        p_action, p_log_prob, p_value = p_out, 0.0, 0.0
                    prompt_obs_list.append(np.asarray(prompt_obs, dtype=np.float32).copy())
                    prompt_actions.append(int(p_action))
                    prompt_log_probs.append(float(p_log_prob))
                    prompt_values.append(float(p_value))
                    prompt_obs, _step_r, done, _trunc, _p_info = prompt_env.step(p_action)
            finally:
                prompt_env.skip_execute = False

            selected = prompt_env.selected_prompts
            # 4. Build prompt suffixes from selected atoms (mirrors PromptEnv._execute_workflow)
            def _atom_text(role: str, idx: int) -> str:
                atoms = library.PROMPT_ATOMS.get(role, {})
                if idx in atoms:
                    return atoms[idx]
                return ""
            reasoner_suffix = "\n".join(_atom_text("reasoner", i) for i in selected["reasoner"] if i != 0)
            verifier_suffix = "\n".join(_atom_text("verifier", i) for i in selected["verifier"] if i != 0)
            answerer_suffix = "\n".join(_atom_text("answerer", i) for i in selected["answerer"] if i != 0)

            # 5. Build configured agent for THIS turn and get one agent action
            workflow = prompt_env.get_workflow_func(workflow_depth, prompt_env.worker, prompt_env.tools)
            if workflow_depth == 2 and hasattr(workflow, "use_verifier"):
                workflow.use_verifier = True
            agent1_tools = prompt_env._decode_tools(agent1_tools_idx)
            agent2_tools = prompt_env._decode_tools(agent2_tools_idx)
            tk_b = prompt_env.TOKEN_BUDGETS
            agent_obj = ConfiguredAgent(
                workflow=workflow,
                worker=prompt_env.worker,
                tool_registry=prompt_env.tools,
                prompt_suffixes={"reasoner": reasoner_suffix, "verifier": verifier_suffix, "answerer": answerer_suffix},
                agent1_tools=agent1_tools,
                agent2_tools=agent2_tools,
                agent1_tokens=tk_b["reasoner"][agent1_budget_idx],
                agent2_tokens=tk_b["verifier"][agent2_budget_idx],
                answerer_tokens=tk_b["answerer"][answerer_budget_idx],
                agent1_budget=agent1_budget_idx,
                agent2_budget=agent2_budget_idx,
                answerer_budget=answerer_budget_idx,
            )
            history = "\n".join(transcript_parts).strip()
            action_internal = agent_obj.respond(history)

            # Token budget guard
            if getattr(agent_obj, "cumulative_tokens", 0) > total_token_cap:
                transcript_parts.append("[system] Token budget exhausted.")
                break

            if _TOOL_LINE_RE.search(action_internal or ""):
                tool_call_attempts += 1
            transcript_parts.append(f"agent: {action_internal}")
            action_tau2 = _action_to_tau2_format(action_internal)

            # 6. Step the env once for this turn
            obs, reward, terminated, truncated, info = env.step(action_tau2)
            shaping_value = float(info.get("shaped_reward", 0.0)) if isinstance(info, dict) else 0.0
            terminal_value = 0.0
            if terminated:
                # ShapedTau2Env.step returns shaping + terminal; back-out the terminal portion
                terminal_value = float(reward) - shaping_value
                last_terminal_reward = terminal_value
            per_turn_rewards.append(shaping_value if not terminated else (shaping_value + terminal_value))
            per_turn_components.append(dict(info.get("shaped_components", {})) if isinstance(info, dict) else {})
            last_info = info or {}

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
    total_reward = float(sum(per_turn_rewards))

    # Diagnostics — reuse rollout's parser for the per-component breakdown
    from tau2_code.src.rollout import _parse_reward_info
    exec_info = {
        "steps": max(len(per_turn_rewards), 1),
        "tools_count": tool_call_attempts,
        "tools_attempted": tool_call_attempts,
        "total_tokens": 0,  # ConfiguredAgent doesn't expose this directly here
        "valid_code_count": 0,
        "file_access_count": 0,
        "tau2_reward": last_terminal_reward,
        "tau2_turns": len(per_turn_rewards),
        "tau2_per_turn_shaping": list(map(float, per_turn_rewards)),
        "tau2_per_turn_components": per_turn_components,
        "tau2_total_shaping": float(sum(c.get("action_progress", 0) + c.get("tool_error", 0) + c.get("tool_dup", 0) + c.get("tokens", 0) for c in per_turn_components)),
        "tau2_shaping_mode": shaping_mode if use_shaped_rewards else "off",
        "tau2_per_turn_config": True,
    }
    exec_info.update(_parse_reward_info(last_info))

    return {
        "structure_obs_list": structure_obs_list,
        "structure_actions": structure_actions,
        "structure_log_probs": structure_log_probs,
        "structure_values": structure_values,
        "structure_action_masks": structure_action_masks,
        "prompt_obs_list": prompt_obs_list,
        "prompt_actions": prompt_actions,
        "prompt_log_probs": prompt_log_probs,
        "prompt_values": prompt_values,
        "per_turn_rewards": per_turn_rewards,
        "per_turn_components": per_turn_components,
        "terminal_reward": float(last_terminal_reward),
        "total_reward": total_reward,
        "transcript": transcript,
        "exec_info": exec_info,
    }
