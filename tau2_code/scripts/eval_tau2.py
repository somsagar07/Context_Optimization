"""Method-agnostic tau2 evaluation driver.

Supports multiple agent types behind the same loop, so we can compare:
  --agent-type base      Plain LLM agent (no policy, no atoms). Closest to tau2's vanilla run.
  --agent-type policy    Use a trained structure+prompt policy checkpoint.

For every task in the split it: builds an agent, runs tau2_dialog_rollout, captures
tau2's per-component breakdown, then aggregates per-domain pass rates.

Saves per-episode JSON + a summary to eval_logs/tau2_<domain>_<agent>_<ts>.{json,summary.txt}.

Usage:
    # Base model on full mock test split (cheap sanity check)
    python tau2_code/scripts/eval_tau2.py --domain mock --agent-type base \
        --agent-llm openai/gpt-4o-mini --episodes all --workers 4

    # Base model on airline test, 20 episodes, parallel
    python tau2_code/scripts/eval_tau2.py --domain airline --agent-type base \
        --agent-llm openai/gpt-4o-mini --episodes 20 --workers 8

    # Trained policy on airline test
    python tau2_code/scripts/eval_tau2.py --domain airline --agent-type policy \
        --structure-model models/ppo/.../structure_..._final.pt \
        --prompt-model models/ppo/.../prompt_..._final.pt \
        --episodes 20 --workers 4
"""
import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO, ".env"))

# Quiet our worker init prints (per-worker "OpenRouter Worker initialized..." etc.) so
# the tqdm bar isn't drowned out when running with --workers > 1. Must be set BEFORE
# importing OpenRouterWorker.
os.environ.setdefault("CO_QUIET", "1")

# Quiet tau2's loguru output and litellm/HTTP logs - eval already saves everything to JSON.
import logging
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
except Exception:
    pass
for noisy in ("LiteLLM", "litellm", "httpx", "openai", "urllib3", "tau2"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

from tqdm import tqdm

from agents_system.worker import OpenRouterWorker
from agents_system.workflows import get_openrouter_workflow

from tau2_code.src import Tau2Dataset, Tau2ToolRegistry, ConfiguredAgent, tau2_dialog_rollout
from tau2_code.src.configured_agent import _build_tool_descriptions_block, _PER_TURN_SYSTEM_INSTRUCTION


# --------------------------------------------------------------------------- #
# Agent variants
# --------------------------------------------------------------------------- #
class BaseAgent:
    """Minimal agent: no policy, no atoms, no workflow.

    Sends conversation_history + tool descriptions + per-turn instruction directly to
    the LLM and emits its response. Equivalent to running gpt-4o-mini as a vanilla
    customer-support agent — the tau2 paper baseline.
    """

    def __init__(self, worker, tool_registry, tokens: int = 1024):
        self.worker = worker
        self.tool_registry = tool_registry
        self.tokens = tokens
        self.cumulative_tokens = 0
        self.cumulative_steps = 0
        self.cumulative_tool_calls = 0

        tool_block = ""
        if hasattr(tool_registry, "get_descriptions_dict"):
            tool_block = _build_tool_descriptions_block(tool_registry.get_descriptions_dict())
        self.system_prompt = "\n\n".join(s for s in (tool_block, _PER_TURN_SYSTEM_INSTRUCTION) if s)

    def respond(self, conversation_history: str) -> str:
        out = self.worker.answer_direct(
            conversation_history,
            tools=[],                                # we inject tool descriptions via prompt_suffix
            tokens=self.tokens,
            prompt_suffix=self.system_prompt,
        )
        self.cumulative_steps += 1
        self.cumulative_tokens += self.tokens
        return (out or "").strip()

    def stats(self) -> Dict[str, int]:
        return {
            "steps": self.cumulative_steps,
            "total_tokens": self.cumulative_tokens,
            "tools_count": self.cumulative_tool_calls,
        }


def build_policy_agent(
    worker, tool_registry, structure_model_path: str, prompt_model_path: str,
    domain: str, task_question: str
):
    """Stub kept for backward compatibility with --agent-type policy.
    The real HRL eval path is --agent-type hrl (see _run_one_task_hrl below)."""
    raise NotImplementedError(
        "--agent-type policy is deprecated. Use --agent-type hrl with "
        "--structure-model and --prompt-model."
    )


# --------------------------------------------------------------------------- #
# HRL agent: drive the trained structure + prompt policies through the same
# Structure/PromptEnv path used at training time. Each task gets policy-selected
# (workflow, agent_tool_groups, budgets, atoms), bound into a ConfiguredAgent,
# then rolled out by tau2_dialog_rollout.
# --------------------------------------------------------------------------- #
class _HRLState:
    """Globals shared across worker threads for --agent-type hrl."""
    structure_policy = None
    prompt_policy = None
    cfg = None
    shared_dataset = None
    api_model = None
    deterministic = True
    temperature = 1.0


_hrl_thread_local = threading.local()


def _get_hrl_envs():
    """Lazy per-thread (StructureEnv, PromptEnv) pair. Each thread gets its own
    pair so we don't need a lock; they share the dataset and the loaded
    OpenRouterWorker is constructed once per thread (not per episode)."""
    pair = getattr(_hrl_thread_local, "envs", None)
    if pair is not None:
        return pair
    from env.structure_env import StructureEnv
    from env.prompt_env import PromptEnv
    cfg = _HRLState.cfg
    se = StructureEnv(
        cfg, is_eval=True, use_api=True,
        api_model=_HRLState.api_model,
        dataset=_HRLState.shared_dataset,
    )
    pe = PromptEnv(
        cfg, is_eval=True, use_api=True,
        api_model=_HRLState.api_model,
        dataset=_HRLState.shared_dataset,
    )
    _hrl_thread_local.envs = (se, pe)
    return se, pe


def _render_tau2_question(task) -> str:
    """Re-render a tau2 task into the same text Tau2Dataset.get_sample() returns,
    so embeddings hit the precomputed cache."""
    from tau2_code.src.dataset import _render_task_description
    return _render_task_description(task)


def _run_one_task_hrl(task_id: str, domain: str, max_turns: int):
    """Run one tau2 episode driven by the loaded HRL policies.

    Returns the same result-dict shape as _run_one_task_body so aggregate() works.
    """
    import torch
    import numpy as np

    structure_env, prompt_env = _get_hrl_envs()

    # Resolve task description deterministically (no random sampling).
    task = structure_env.dataset._task_by_id[task_id]
    question = _render_tau2_question(task)

    # Set the task on the structure env without going through reset() (which would
    # randomize). Mirrors the pattern in scripts/eval_hrl.run_single_episode.
    structure_env.current_q = question
    structure_env.current_a = task_id  # tau2 dataset stores task_id in the answer slot
    structure_env.question_embedding = structure_env.worker.get_embedding(question)
    struct_obs = structure_env._get_observation()

    # Structure policy decision
    with torch.no_grad():
        struct_action = _HRLState.structure_policy.get_action(
            struct_obs,
            deterministic=_HRLState.deterministic,
            temperature=_HRLState.temperature,
        )
    if not isinstance(struct_action, np.ndarray):
        struct_action = np.array(struct_action, dtype=np.int64)
    _, _, _, _, struct_info = structure_env.step(struct_action)

    # Prompt env setup
    prompt_env.set_structure(
        question=struct_info["question"],
        answer=struct_info["answer"],
        embedding=struct_info["embedding"],
        structure=struct_info["structure"],
    )
    prompt_obs, _ = prompt_env.reset()

    # Roll out the prompt policy. The final step triggers _execute_tau2_workflow
    # (which calls tau2_dialog_rollout) inside the env.
    done = False
    info = {}
    t0 = time.time()
    try:
        while not done:
            with torch.no_grad():
                action = _HRLState.prompt_policy.get_action(
                    prompt_obs,
                    deterministic=_HRLState.deterministic,
                    temperature=_HRLState.temperature,
                )
            prompt_obs, _step_reward, done, _, info = prompt_env.step(action)
        wall = time.time() - t0
    except Exception as e:
        return {
            "task_id": task_id,
            "domain": domain,
            "agent_type": "hrl",
            "agent_llm": _HRLState.api_model,
            "error": f"{type(e).__name__}: {e}",
            "tau2_reward": 0.0,
            "binary_pass": False,
        }

    paper_reward = float(info.get("tau2_original_reward", info.get("correctness", 0.0)))
    shaped_reward = float(info.get("correctness", paper_reward))
    return {
        "task_id": task_id,
        "domain": domain,
        "agent_type": "hrl",
        "agent_llm": _HRLState.api_model,
        "tau2_reward": paper_reward,
        "tau2_reward_shaped": shaped_reward,
        "binary_pass": paper_reward >= 0.999,
        "wall_seconds": round(wall, 2),
        "tau2_action_pass":      info.get("tau2_action_pass", 0),
        "tau2_action_total":     info.get("tau2_action_total", 0),
        "tau2_action_missed":    info.get("tau2_action_missed", 0),
        "tau2_communicate_pass":   info.get("tau2_communicate_pass", 0),
        "tau2_communicate_total":  info.get("tau2_communicate_total", 0),
        "tau2_communicate_missed": info.get("tau2_communicate_missed", 0),
        "tau2_env_pass":   info.get("tau2_env_pass", 0),
        "tau2_env_total":  info.get("tau2_env_total", 0),
        "tau2_env_missed": info.get("tau2_env_missed", 0),
        "tau2_nl_pass":   info.get("tau2_nl_pass", 0),
        "tau2_nl_total":  info.get("tau2_nl_total", 0),
        "tau2_nl_missed": info.get("tau2_nl_missed", 0),
        "tau2_reward_basis":       info.get("tau2_reward_basis", []),
        "tau2_termination_reason": info.get("tau2_termination_reason", ""),
        "tau2_turns":              info.get("tau2_turns", 0),
        "agent_total_tokens": info.get("total_tokens", 0),
        "agent_steps":        info.get("steps_taken", 0),
        "workflow":           info.get("workflow", ""),
        "agent1_tools":       struct_info["structure"].get("agent1_tools_idx"),
        "agent2_tools":       struct_info["structure"].get("agent2_tools_idx"),
    }


# --------------------------------------------------------------------------- #
# Per-episode runner
# --------------------------------------------------------------------------- #
_log_lock = threading.Lock()

# Live counter of how many tasks are currently mid-execution on worker threads.
# Incremented at the start of run_one_task; decremented on exit (try/finally).
# Read by the tqdm postfix to show "ongoing=X/W".
_inflight_lock = threading.Lock()
_inflight = 0


def run_one_task(
    task_id: str, domain: str, agent_type: str, agent_llm: str, max_turns: int,
    solo_mode: bool, agent_tokens: int, trial_idx: int = 0,
):
    """Run one tau2 episode end-to-end and return a structured result dict.
    `trial_idx` identifies which independent trial this is (0..num_trials-1) for
    a given task; user simulator stochasticity gives different conversations across trials."""
    global _inflight
    with _inflight_lock:
        _inflight += 1
    try:
        result = _run_one_task_body(
            task_id, domain, agent_type, agent_llm, max_turns, solo_mode, agent_tokens,
        )
        result["trial_idx"] = trial_idx
        return result
    finally:
        with _inflight_lock:
            _inflight -= 1


def _run_one_task_body(
    task_id: str, domain: str, agent_type: str, agent_llm: str, max_turns: int,
    solo_mode: bool, agent_tokens: int,
):
    # HRL path goes through the StructureEnv/PromptEnv pipeline, not the simple
    # rollout below. Dispatch early so we don't construct an unused worker.
    if agent_type == "hrl":
        return _run_one_task_hrl(task_id, domain=domain, max_turns=max_turns)

    worker = OpenRouterWorker(model_name=agent_llm)
    tools = Tau2ToolRegistry(domain=domain)

    if agent_type == "base":
        agent = BaseAgent(worker, tools, tokens=agent_tokens)
    else:
        raise ValueError(f"Unsupported agent_type for eval_tau2.py: {agent_type}")

    t0 = time.time()
    try:
        transcript, exec_info, reward = tau2_dialog_rollout(
            agent, domain=domain, task_id=task_id,
            max_turns=max_turns, solo_mode=solo_mode,
            use_shaped_rewards=True, shaping_mode="eval",
        )
        wall = time.time() - t0
        paper_reward = float(exec_info.get("tau2_original_reward", reward))
        return {
            "task_id": task_id,
            "domain": domain,
            "agent_type": agent_type,
            "agent_llm": agent_llm,
            "tau2_reward": paper_reward,
            "tau2_reward_shaped": float(reward),
            "binary_pass": paper_reward >= 0.999,
            "wall_seconds": round(wall, 2),
            "tau2_action_pass": exec_info.get("tau2_action_pass", 0),
            "tau2_action_total": exec_info.get("tau2_action_total", 0),
            "tau2_action_missed": exec_info.get("tau2_action_missed", 0),
            "tau2_communicate_pass": exec_info.get("tau2_communicate_pass", 0),
            "tau2_communicate_total": exec_info.get("tau2_communicate_total", 0),
            "tau2_communicate_missed": exec_info.get("tau2_communicate_missed", 0),
            "tau2_env_pass": exec_info.get("tau2_env_pass", 0),
            "tau2_env_total": exec_info.get("tau2_env_total", 0),
            "tau2_env_missed": exec_info.get("tau2_env_missed", 0),
            "tau2_nl_pass": exec_info.get("tau2_nl_pass", 0),
            "tau2_nl_total": exec_info.get("tau2_nl_total", 0),
            "tau2_nl_missed": exec_info.get("tau2_nl_missed", 0),
            "tau2_reward_basis": exec_info.get("tau2_reward_basis", []),
            "tau2_termination_reason": exec_info.get("tau2_termination_reason", ""),
            "tau2_turns": exec_info.get("tau2_turns", 0),
            "agent_total_tokens": exec_info.get("total_tokens", 0),
            "agent_steps": exec_info.get("steps", 0),
            "transcript_tail": transcript[-2000:],     # truncate for log size
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "domain": domain,
            "agent_type": agent_type,
            "agent_llm": agent_llm,
            "error": f"{type(e).__name__}: {e}",
            "tau2_reward": 0.0,
            "binary_pass": False,
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Tau2-bench evaluation driver.")
    p.add_argument("--domain", required=True, choices=["mock", "airline", "retail", "telecom"])
    p.add_argument("--split", default="test", choices=["train", "test", "base"])
    p.add_argument("--episodes", default="all", help="'all' or an integer")
    p.add_argument("--workers", type=int, default=1, help="parallel API workers (>=1)")
    p.add_argument("--agent-type", default="base", choices=["base", "policy", "hrl"])
    p.add_argument("--agent-llm", default="openai/gpt-4o-mini",
                   help="Bare OpenRouter model id (no 'openrouter/' prefix) for our agent")
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--agent-tokens", type=int, default=1024,
                   help="Per-turn token budget for the base agent")
    p.add_argument("--solo", action="store_true", help="Solo mode (no user simulator)")
    p.add_argument("--num-trials", type=int, default=1,
                   help="Number of independent trials per task (paper uses k=4 for pass^k). "
                        "Each trial is a fresh dialog with a re-rolled user simulator. "
                        "When >1, summary reports pass^1..pass^k.")
    p.add_argument("--out-dir", default="eval_logs")

    # HRL-specific (only used with --agent-type hrl).
    p.add_argument("--structure-model", default=None,
                   help="Path to a trained structure-policy checkpoint (required for --agent-type hrl).")
    p.add_argument("--prompt-model", default=None,
                   help="Path to a trained prompt-policy checkpoint (required for --agent-type hrl).")
    p.add_argument("--api-model", default=None,
                   help="OpenRouter model id used by the HRL agent (defaults to --agent-llm).")
    p.add_argument("--config", default="hierarchical",
                   help="HRL config to load (default: hierarchical).")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample policy actions instead of taking argmax (HRL eval).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Policy softmax temperature (HRL eval, only with --stochastic).")
    return p.parse_args()


def aggregate(results: List[Dict], num_trials: int = 1) -> Dict:
    """Aggregate per-trial results into:
      - per-trial mean metrics (avg reward, turns, tokens, pct_*) — pooled across all trials
      - pass^k curve when num_trials > 1: pass^j for j = 1..num_trials.

    pass^j for a task = 1 iff every one of the first j trials of that task scored
    binary_pass=True. Final pass^j is the mean over tasks.
    """
    from collections import defaultdict

    total_trials = len(results)
    succ = [r for r in results if not r.get("error")]
    avg_reward = (sum(r.get("tau2_reward", 0.0) for r in succ) / max(len(succ), 1))
    avg_reward_shaped = (sum(r.get("tau2_reward_shaped", 0.0) for r in succ) / max(len(succ), 1))
    avg_turns = (sum(r.get("tau2_turns", 0) for r in succ) / max(len(succ), 1))
    avg_tokens = (sum(r.get("agent_total_tokens", 0) for r in succ) / max(len(succ), 1))

    def _pct(num_field, denom_field):
        num = sum(r.get(num_field, 0) for r in succ)
        den = sum(r.get(denom_field, 0) for r in succ)
        return (num / den * 100.0) if den > 0 else None

    # Group by task_id, sort by trial_idx, compute pass^k curve.
    by_task = defaultdict(list)
    for r in results:
        by_task[r.get("task_id")].append(r)
    for tid in list(by_task):
        by_task[tid].sort(key=lambda r: r.get("trial_idx", 0))

    n_tasks = len(by_task)
    pass_curve = {}     # "pass^1", "pass^2", ...
    if n_tasks > 0:
        for j in range(1, num_trials + 1):
            n_pass_at_j = 0
            for tid, trials in by_task.items():
                if len(trials) < j:
                    continue
                all_pass = all(bool(t.get("binary_pass")) for t in trials[:j])
                if all_pass:
                    n_pass_at_j += 1
            pass_curve[f"pass^{j}"] = n_pass_at_j / n_tasks

    # pass^1 (binary pass rate) is the standard headline number even at num_trials=1
    headline_pass = pass_curve.get("pass^1")
    if headline_pass is None:
        # Fallback: trial-pooled rate
        bin_pass = sum(1 for r in succ if r.get("binary_pass"))
        headline_pass = bin_pass / max(total_trials, 1)

    out = {
        "num_trials": num_trials,
        "n_tasks": n_tasks,
        "total_trials": total_trials,
        "errors": total_trials - len(succ),
        "binary_pass_rate": headline_pass,        # = pass^1 when num_trials >= 1
        "avg_tau2_reward": avg_reward,
        "avg_tau2_reward_shaped": avg_reward_shaped,
        "avg_turns": avg_turns,
        "avg_agent_tokens": avg_tokens,
        "pct_action_pass":      _pct("tau2_action_pass",      "tau2_action_total"),
        "pct_communicate_pass": _pct("tau2_communicate_pass", "tau2_communicate_total"),
        "pct_env_pass":         _pct("tau2_env_pass",         "tau2_env_total"),
        "pct_nl_pass":          _pct("tau2_nl_pass",          "tau2_nl_total"),
    }
    out.update(pass_curve)
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load split
    ds = Tau2Dataset(domain=args.domain, split=args.split)
    task_ids = [t.id for t in ds.tasks]
    if args.episodes != "all":
        try:
            n = int(args.episodes)
        except ValueError:
            raise SystemExit(f"--episodes must be 'all' or an integer, got {args.episodes!r}")
        task_ids = task_ids[:n]

    # HRL setup: load policies + atoms once, then per-thread envs come up lazily.
    if args.agent_type == "hrl":
        if not args.structure_model or not args.prompt_model:
            raise SystemExit(
                "--agent-type hrl requires --structure-model and --prompt-model paths."
            )
        # Reuse the policy classes/loaders from scripts/eval_hrl.py.
        from scripts.eval_hrl import load_structure_policy, load_prompt_policy
        from configs import load_config
        from prompts import library

        cfg = load_config(args.config)
        cfg.DATASET_NAME = f"tau2_{args.domain}"
        cfg.TAU2_MAX_TURNS = int(args.max_turns)
        cfg.USE_SHAPED_REWARDS = True
        cfg.SHAPING_MODE = "eval"
        cfg.SHAPING_CFG = None

        # Load atoms cache so PromptEnv sizes its action space the same way it did
        # at training time (otherwise the policy's atom indices may exceed action_dim
        # and get clamped to DONE).
        atoms_path = library._get_atoms_path(cfg.DATASET_NAME)
        if os.path.exists(atoms_path):
            library.load_or_create_atoms(cfg.DATASET_NAME, worker=None)
        else:
            print(f"  ⚠ No atoms cache at {atoms_path}; using base atoms only.")

        # Use CPU for the policies. They're tiny MLPs (~256 hidden) so GPU buys
        # nothing, AND it sidesteps a torch.load failure mode where
        # `cuda.is_available()` returns True but `cuda.device_count()` is 0 (e.g.
        # CUDA_VISIBLE_DEVICES set to a non-existent index). The match-eval_hrl
        # device guard is also fine, but CPU is the simplest correct default here.
        import torch as _torch
        if _torch.cuda.is_available() and _torch.cuda.device_count() > 0:
            device = "cuda"
        else:
            device = "cpu"
        struct_policy, struct_algo = load_structure_policy(args.structure_model, device)
        prmpt_policy, prmpt_algo = load_prompt_policy(args.prompt_model, device)

        _HRLState.structure_policy = struct_policy
        _HRLState.prompt_policy = prmpt_policy
        _HRLState.cfg = cfg
        _HRLState.shared_dataset = ds
        _HRLState.api_model = args.api_model or args.agent_llm
        _HRLState.deterministic = not args.stochastic
        _HRLState.temperature = float(args.temperature)
        print(f"[eval] HRL policies loaded ({struct_algo}/{prmpt_algo}); "
              f"api_model={_HRLState.api_model} deterministic={_HRLState.deterministic}")

    n_trials = max(1, int(args.num_trials))
    work = [(tid, j) for tid in task_ids for j in range(n_trials)]
    print(f"[eval] domain={args.domain} split={args.split} agent={args.agent_type}/{args.agent_llm} "
          f"tasks={len(task_ids)} trials/task={n_trials} total_runs={len(work)} workers={args.workers}")

    # Run with a tqdm progress bar showing live pass-rate / avg reward.
    results: List[Dict] = []
    t0 = time.time()
    bar = tqdm(total=len(work), desc=f"eval {args.domain}", unit="run", dynamic_ncols=True)

    def _on_done(r):
        with _log_lock:
            results.append(r)
            n = len(results)
            passes = sum(1 for x in results if x.get("binary_pass"))
            avg_r = sum(x.get("tau2_reward", 0.0) for x in results) / max(n, 1)
            with _inflight_lock:
                inflight_now = _inflight
            postfix = {
                "ongoing": f"{inflight_now}/{args.workers}",
                "pass^1": f"{passes}/{n} ({passes/max(n,1)*100:.1f}%)",
                "avg_reward": f"{avg_r:.3f}",
                "last": "OK" if r.get("binary_pass") else f"r={r.get('tau2_reward',0):.2f}",
            }
            if n_trials > 1:
                postfix["trial"] = f"{r.get('task_id')}#{r.get('trial_idx', 0)+1}/{n_trials}"
            bar.set_postfix(postfix)
            bar.update(1)

    if args.workers <= 1:
        for tid, trial_idx in work:
            r = run_one_task(tid, args.domain, args.agent_type, args.agent_llm,
                             args.max_turns, args.solo, args.agent_tokens, trial_idx=trial_idx)
            _on_done(r)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one_task, tid, args.domain, args.agent_type, args.agent_llm,
                            args.max_turns, args.solo, args.agent_tokens, trial_idx): (tid, trial_idx)
                for tid, trial_idx in work
            }
            for fut in as_completed(futures):
                _on_done(fut.result())
    bar.close()

    wall = time.time() - t0
    summary = aggregate(results, num_trials=n_trials)
    summary["wall_seconds"] = round(wall, 1)

    # Persist with a meaningful directory structure:
    #   eval_logs/tau2/<domain>/<agent_type>/<llm_slug>/<timestamp>/
    #     args.json         -- CLI args used for the run
    #     summary.txt       -- human-readable aggregated metrics
    #     summary.json      -- machine-readable aggregated metrics
    #     results.json      -- full per-task records (truncated transcripts inline)
    #     transcripts/      -- per-task full transcripts, one file per task
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_slug = args.agent_llm.replace("/", "-")
    run_dir = os.path.join(
        args.out_dir, "tau2", args.domain, args.agent_type, llm_slug, ts
    )
    transcripts_dir = os.path.join(run_dir, "transcripts")
    os.makedirs(transcripts_dir, exist_ok=True)

    # args.json — exact CLI invocation (for reproducibility)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # summary.json — machine-readable aggregates
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # results.json — full per-task records, with the transcript_tail kept inline.
    # Full transcripts go to per-task files in transcripts/ for easier diffing.
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Per-task-per-trial transcript files. With num_trials > 1 we need a unique name per trial.
    for r in results:
        tid = str(r.get("task_id", "unknown")).replace("/", "_").replace(" ", "_")
        trial_idx = int(r.get("trial_idx", 0))
        suffix = f"_trial{trial_idx}" if n_trials > 1 else ""
        path = os.path.join(transcripts_dir, f"{tid}{suffix}.txt")
        body_lines = [
            f"task_id: {r.get('task_id')}",
            f"domain: {r.get('domain')}",
            f"agent_type: {r.get('agent_type')}",
            f"agent_llm: {r.get('agent_llm')}",
            f"tau2_reward: {r.get('tau2_reward')}",
            f"binary_pass: {r.get('binary_pass')}",
            f"reward_basis: {r.get('tau2_reward_basis')}",
            f"action: {r.get('tau2_action_pass')}/{r.get('tau2_action_total')}  "
            f"(missed={r.get('tau2_action_missed')})",
            f"communicate: {r.get('tau2_communicate_pass')}/{r.get('tau2_communicate_total')}  "
            f"(missed={r.get('tau2_communicate_missed')})",
            f"env: {r.get('tau2_env_pass')}/{r.get('tau2_env_total')}  "
            f"(missed={r.get('tau2_env_missed')})",
            f"nl: {r.get('tau2_nl_pass')}/{r.get('tau2_nl_total')}  "
            f"(missed={r.get('tau2_nl_missed')})",
            f"termination: {r.get('tau2_termination_reason')}",
            f"turns: {r.get('tau2_turns')}",
            f"agent_tokens: {r.get('agent_total_tokens')}",
            f"wall_seconds: {r.get('wall_seconds')}",
            "-" * 60,
            "transcript (tail):",
            r.get("transcript_tail", "") or r.get("error", ""),
        ]
        with open(path, "w") as f:
            f.write("\n".join(body_lines) + "\n")

    # summary.txt — what the user actually reads.
    # Print pass^k curve prominently when num_trials > 1.
    lines = [
        "=" * 60,
        f"Tau2 eval: domain={args.domain} split={args.split}",
        f"Agent:    type={args.agent_type} llm={args.agent_llm}",
        f"Tasks:    {len(task_ids)}  Trials/task: {n_trials}  Total runs: {len(results)}",
        f"Workers:  {args.workers}  Wall: {wall:.1f}s",
        f"Run dir:  {run_dir}",
        "-" * 60,
    ]
    if n_trials > 1:
        lines.append("  pass^k curve (paper metric):")
        for j in range(1, n_trials + 1):
            v = summary.get(f"pass^{j}")
            if v is not None:
                lines.append(f"    pass^{j}: {v:.4f}  ({int(round(v * summary['n_tasks']))}/{summary['n_tasks']} tasks)")
        lines.append("-" * 60)
    for k, v in summary.items():
        if k.startswith("pass^"):
            continue  # already shown above
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}" if abs(v) < 1000 else f"  {k}: {v:.2f}")
        else:
            lines.append(f"  {k}: {v}")
    lines.append("=" * 60)
    summary_text = "\n".join(lines)
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    print()
    print(summary_text)
    print(f"\nResults written under: {run_dir}/")
    print(f"  args.json       summary.txt    summary.json")
    print(f"  results.json    transcripts/   ({len(results)} files)")


if __name__ == "__main__":
    main()
