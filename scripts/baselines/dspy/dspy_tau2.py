"""DSPy baseline for tau2-bench.

Simple approach: DSPy (MIPROv2) optimizes the system instruction for a base LLM
acting as a customer-support agent. The agent uses litellm directly (no workflow
machinery) with tool descriptions injected into the prompt.

Usage:
    # Baseline (no optimization, raw model)
    python scripts/baselines/dspy/dspy_tau2.py baseline --domain telecom --n-eval 5

    # Train (optimize instructions via MIPROv2)
    python scripts/baselines/dspy/dspy_tau2.py train --domain telecom --n-train 10 --n-dev 5

    # Evaluate saved optimized instruction
    python scripts/baselines/dspy/dspy_tau2.py eval --domain telecom \
        --prompt-path gen_prompts/dspy_tau2/telecom_light.json --n-eval 20
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO, ".env"))

os.environ.setdefault("LITELLM_LOG", "ERROR")

import logging
import warnings
warnings.filterwarnings("ignore")
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
except Exception:
    pass
for _noisy in ("LiteLLM", "litellm", "httpx", "openai", "urllib3", "tau2",
               "gymnasium", "dspy", "transformers"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False


# ---------------------------------------------------------------------------
# Tool description builder
# ---------------------------------------------------------------------------

def get_tool_descriptions(domain: str) -> str:
    """Get formatted tool descriptions for a tau2 domain."""
    from tau2_code.src.tool_registry import Tau2ToolRegistry
    registry = Tau2ToolRegistry(domain=domain)
    descs = registry.get_descriptions_dict()
    lines = ["Available tools (use format: TOOL: <name> || QUERY: <json_args>):"]
    for name, desc in descs.items():
        lines.append(f"\n{desc}")
    return "\n".join(lines)


def get_tool_names(domain: str) -> List[str]:
    """Get list of all tool names for a domain."""
    from tau2_code.src.tool_registry import Tau2ToolRegistry
    registry = Tau2ToolRegistry(domain=domain)
    return registry.list_tools()


# ---------------------------------------------------------------------------
# Simple LLM agent for tau2 dialogs
# ---------------------------------------------------------------------------

class SimpleLLMAgent:
    """Minimal agent: system prompt + conversation history -> next action via litellm."""

    def __init__(self, model: str, system_prompt: str, max_tokens: int = 1024):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages = [{"role": "system", "content": system_prompt}]
        self.total_tokens = 0

    def respond(self, observation: str) -> str:
        """Given new observation text, generate next response."""
        if observation:
            self.messages.append({"role": "user", "content": observation})

        try:
            response = litellm.completion(
                model=self.model,
                messages=self.messages,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            reply = response.choices[0].message.content or ""
            self.total_tokens += response.usage.total_tokens if response.usage else 0
            self.messages.append({"role": "assistant", "content": reply})
            return reply.strip()
        except Exception as e:
            error_msg = f"[Agent Error: {e}]"
            self.messages.append({"role": "assistant", "content": error_msg})
            return error_msg


# ---------------------------------------------------------------------------
# Tau2 dialog runner
# ---------------------------------------------------------------------------

_TOOL_RE = re.compile(r"TOOL:\s*([A-Za-z_]\w*)\s*\|\|\s*QUERY:\s*(.+)", re.DOTALL)


def _action_to_tau2(action_text: str) -> str:
    """Convert 'TOOL: name || QUERY: {args}' to tau2 JSON format, or pass plain text."""
    if not action_text:
        return ""
    m = _TOOL_RE.search(action_text)
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


def run_episode(
    domain: str,
    task_id: str,
    system_prompt: str,
    model: str,
    max_turns: int = 12,
) -> Dict:
    """Run one tau2 episode with a simple LLM agent. Returns result dict."""
    import gymnasium as gym
    from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        register_gym_agent()

    user_llm = os.environ.get("OPENROUTER_USER_MODEL", "openrouter/openai/gpt-4o-mini")

    env = gym.make(
        TAU_BENCH_ENV_ID,
        domain=domain,
        task_id=task_id,
        solo_mode=False,
        max_steps=max(2 * max_turns, 8),
        user_llm=user_llm,
    )

    agent = SimpleLLMAgent(model=model, system_prompt=system_prompt)
    transcript_parts = []
    tool_calls = 0

    try:
        obs, info = env.reset()
        if not obs:
            return {
                "task_id": task_id, "reward": 0.0, "turns": 0,
                "transcript": "", "termination_reason": "sim_failed",
                "tools_count": 0, "total_tokens": 0,
            }
        transcript_parts.append(f"user: {obs}")

        for turn in range(max_turns):
            action = agent.respond(obs)
            transcript_parts.append(f"agent: {action}")

            if _TOOL_RE.search(action or ""):
                tool_calls += 1

            action_tau2 = _action_to_tau2(action)
            obs, reward, terminated, truncated, info = env.step(action_tau2)

            if obs:
                transcript_parts.append(f"{'tool' if tool_calls else 'user'}: {obs}")

            if terminated or truncated:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Parse reward info
    reward_info = {}
    if info and "reward_info" in info:
        try:
            ri = info["reward_info"]
            if isinstance(ri, str):
                ri = json.loads(ri)
            reward_info = ri
        except Exception:
            pass

    # Extract component scores
    from tau2_code.src.rollout import _parse_reward_info
    parsed = _parse_reward_info(info or {})

    result = {
        "task_id": task_id,
        "reward": float(reward) if reward else 0.0,
        "turns": turn + 1 if 'turn' in dir() else 0,
        "transcript": "\n".join(transcript_parts),
        "termination_reason": parsed.get("tau2_termination_reason", ""),
        "tools_count": tool_calls,
        "total_tokens": agent.total_tokens,
        "action_pass": parsed.get("tau2_action_pass", 0),
        "action_total": parsed.get("tau2_action_total", 0),
        "communicate_pass": parsed.get("tau2_communicate_pass", 0),
        "communicate_total": parsed.get("tau2_communicate_total", 0),
        "nl_pass": parsed.get("tau2_nl_pass", 0),
        "nl_total": parsed.get("tau2_nl_total", 0),
    }
    return result


# ---------------------------------------------------------------------------
# Build system prompt
# ---------------------------------------------------------------------------

DEFAULT_INSTRUCTION = (
    "You are a helpful customer-support agent. Your goal is to resolve the "
    "customer's issue efficiently through the available tools.\n\n"
    "Guidelines:\n"
    "- Ask for the customer's identifier (phone, email, name) if not provided\n"
    "- Use tools to look up account info and diagnose issues\n"
    "- Take action to resolve the issue using the appropriate tool\n"
    "- Confirm resolution with the customer\n"
    "- If you cannot resolve after 2-3 attempts, transfer to a human agent\n"
    "- Keep responses concise and action-oriented\n"
    "- Use format: TOOL: <tool_name> || QUERY: {\"arg\": \"value\"} for tool calls"
)


def build_system_prompt(domain: str, instruction: str = None) -> str:
    """Build full system prompt = instruction + tool descriptions."""
    instr = instruction or DEFAULT_INSTRUCTION
    tools = get_tool_descriptions(domain)
    return f"{instr}\n\n{tools}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_tasks(domain: str, split: str = "train", n: Optional[int] = None) -> List[Dict]:
    """Load tau2 tasks."""
    from tau2_code.src.dataset import Tau2Dataset
    dataset = Tau2Dataset(domain=domain, split=split)
    tasks = [{"task_id": item["task_id"], "description": item["question"]} for item in dataset.data]
    if n is not None and n < len(tasks):
        import random
        random.seed(42)
        tasks = random.sample(tasks, n)
    return tasks


def evaluate(
    domain: str,
    model: str,
    instruction: str = None,
    tasks: List[Dict] = None,
    n_eval: Optional[int] = None,
    split: str = "train",
    max_turns: int = 12,
    workers: int = 4,
    num_trials: int = 1,
) -> Dict:
    """Run evaluation on tau2 tasks with pass^k metric.

    When num_trials > 1, each task is run num_trials times independently and
    pass^k is computed using the combinatorial formula from the tau2 paper:
      pass^k(task) = C(successes, k) / C(num_trials, k)
    Final pass^k is the mean over tasks.
    """
    from collections import defaultdict
    from math import comb
    from tqdm import tqdm

    if tasks is None:
        tasks = load_tasks(domain, split=split, n=n_eval)

    system_prompt = build_system_prompt(domain, instruction)
    results = []

    # Build trial jobs: each task repeated num_trials times
    jobs = []
    for task in tasks:
        for trial in range(num_trials):
            jobs.append((task, trial))

    def run_one(job):
        task, trial = job
        result = run_episode(
            domain=domain,
            task_id=task["task_id"],
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
        )
        result["trial"] = trial
        result["binary_pass"] = result.get("reward", 0.0) > 0.5
        return result

    total_jobs = len(jobs)
    if workers <= 1:
        for job in tqdm(jobs, desc="Evaluating"):
            try:
                results.append(run_one(job))
            except Exception as e:
                task, trial = job
                print(f"  Error on {task['task_id']} trial {trial}: {e}")
                results.append({"task_id": task["task_id"], "trial": trial, "reward": 0.0, "binary_pass": False, "error": str(e)})
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, job): job for job in jobs}
            with tqdm(total=total_jobs, desc=f"Evaluating ({num_trials} trial{'s' if num_trials > 1 else ''})") as pbar:
                for future in as_completed(futures):
                    job = futures[future]
                    task, trial = job
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"  Error on {task['task_id']} trial {trial}: {e}")
                        results.append({"task_id": task["task_id"], "trial": trial, "reward": 0.0, "binary_pass": False, "error": str(e)})
                    pbar.update(1)
                    valid = [r for r in results if "error" not in r]
                    if valid:
                        avg_r = sum(r["reward"] for r in valid) / len(valid)
                        corr = sum(1 for r in valid if r.get("binary_pass"))
                        pbar.set_postfix(avg_rew=f"{avg_r:.3f}", correct=f"{corr}/{len(valid)}")

    valid = [r for r in results if "error" not in r]
    avg_reward = sum(r["reward"] for r in valid) / len(valid) if valid else 0.0
    correct = sum(1 for r in valid if r.get("binary_pass"))

    # Compute pass^k curve
    by_task = defaultdict(list)
    for r in results:
        by_task[r.get("task_id")].append(r)

    n_tasks = len(by_task)
    pass_curve = {}
    if n_tasks > 0:
        for k in range(1, num_trials + 1):
            task_scores = []
            for tid, trials in by_task.items():
                n_t = len(trials)
                if n_t < k:
                    continue
                successes = sum(1 for t in trials if bool(t.get("binary_pass")))
                task_scores.append(comb(successes, k) / comb(n_t, k))
            pass_curve[f"pass^{k}"] = (sum(task_scores) / len(task_scores)) if task_scores else 0.0

    headline_pass = pass_curve.get("pass^1", correct / max(len(valid), 1))

    summary = {
        "domain": domain,
        "model": model,
        "instruction": (instruction or DEFAULT_INSTRUCTION)[:500],
        "split": split,
        "num_tasks": n_tasks,
        "num_trials": num_trials,
        "total_trials": len(results),
        "num_completed": len(valid),
        "avg_reward": avg_reward,
        "correct": correct,
        "binary_pass_rate": headline_pass,
        "accuracy_pct": headline_pass * 100,
        "pass_curve": pass_curve,
        "avg_turns": sum(r.get("turns", 0) for r in valid) / len(valid) if valid else 0,
        "avg_tokens": sum(r.get("total_tokens", 0) for r in valid) / len(valid) if valid else 0,
        "timestamp": datetime.now().isoformat(),
        "episodes": results,
    }
    return summary


# ---------------------------------------------------------------------------
# DSPy MIPROv2 optimization
# ---------------------------------------------------------------------------

def train_dspy(args):
    """Optimize system instruction via DSPy MIPROv2."""
    import dspy

    print(f"=== DSPy Tau2 Train: domain={args.domain}, model={args.task_model} ===")

    tasks = load_tasks(args.domain, split="train", n=args.n_train + args.n_dev)
    train_tasks = tasks[:args.n_train]
    dev_tasks = tasks[args.n_train:args.n_train + args.n_dev]
    print(f"  Train: {len(train_tasks)}, Dev: {len(dev_tasks)}")

    # DSPy signature: given a task description, produce agent instructions
    class AgentPromptSignature(dspy.Signature):
        """Generate optimal system instructions for a customer-support agent handling multi-turn dialogs with tool access."""
        task_context: str = dspy.InputField(desc="Description of the customer support domain and typical scenarios")
        agent_prompt: str = dspy.OutputField(desc="System instructions for the agent")

    class AgentPromptOptimizer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(AgentPromptSignature)

        def forward(self, task_context: str) -> dspy.Prediction:
            return self.generate(task_context=task_context)

    # Build domain context from a few task descriptions
    domain_context = f"Domain: {args.domain}\n\nExample tasks:\n"
    for t in train_tasks[:3]:
        domain_context += f"- {t['description'][:200]}\n"
    domain_context += f"\nAvailable tools: {', '.join(get_tool_names(args.domain))}"

    # DSPy examples
    train_examples = [
        dspy.Example(task_context=domain_context).with_inputs("task_context")
        for _ in train_tasks
    ]
    dev_examples = [
        dspy.Example(task_context=domain_context).with_inputs("task_context")
        for _ in dev_tasks
    ]

    # Metric: generate instruction -> run N episodes -> average reward
    def metric_fn(example, prediction, trace=None) -> float:
        instr = getattr(prediction, "agent_prompt", "") or ""
        if not instr:
            return 0.0
        # Run on a single random dev task for speed during optimization
        import random
        task = random.choice(dev_tasks)
        try:
            result = run_episode(
                domain=args.domain,
                task_id=task["task_id"],
                system_prompt=build_system_prompt(args.domain, instr),
                model=args.task_model,
                max_turns=args.max_turns,
            )
            return max(0.0, min(1.0, result["reward"]))
        except Exception as e:
            print(f"  [metric error] {e}")
            return 0.0

    # Configure DSPy
    task_lm = dspy.LM(
        f"openrouter/{args.task_model}" if not args.task_model.startswith("openrouter/") else args.task_model,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
    )
    prompt_lm = dspy.LM(
        f"openrouter/{args.prompt_model}" if not args.prompt_model.startswith("openrouter/") else args.prompt_model,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
    )
    dspy.configure(lm=task_lm)

    solver = AgentPromptOptimizer()

    print(f"  Running MIPROv2 optimization (auto={args.auto})...")
    optimizer = dspy.MIPROv2(
        metric=metric_fn,
        auto=args.auto,
        prompt_model=prompt_lm,
        task_model=task_lm,
    )

    optimized = optimizer.compile(
        student=solver,
        trainset=train_examples,
        valset=dev_examples,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
    )

    # Extract optimized instruction
    opt_instruction = ""
    try:
        state_str = optimized.save()
        state = json.loads(state_str) if isinstance(state_str, str) else state_str
        for key, val in state.items():
            if isinstance(val, dict) and "signature" in val:
                opt_instruction = val["signature"].get("instructions", "")
                if opt_instruction:
                    break
    except Exception as e:
        print(f"  Warning: could not extract instruction from state: {e}")

    if not opt_instruction:
        # Try direct attribute access
        try:
            opt_instruction = str(optimized.generate.predict.signature.instructions)
        except Exception:
            pass

    print(f"\n  Optimized instruction ({len(opt_instruction)} chars):")
    print(f"  {opt_instruction[:300]}...")

    # Save
    out_dir = os.path.join(_REPO, "gen_prompts", "dspy_tau2")
    os.makedirs(out_dir, exist_ok=True)
    model_safe = args.task_model.replace("/", "_")
    out_path = os.path.join(out_dir, f"{args.domain}_{args.auto}_{model_safe}.json")

    save_data = {
        "domain": args.domain,
        "task_model": args.task_model,
        "prompt_model": args.prompt_model,
        "auto": args.auto,
        "n_train": len(train_tasks),
        "n_dev": len(dev_tasks),
        "optimized_instruction": opt_instruction,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        save_data["dspy_state"] = optimized.save()
    except Exception:
        pass

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved to: {out_path}")

    # Post-optimization eval
    if opt_instruction:
        print(f"\n  Post-opt eval on {len(dev_tasks)} dev tasks...")
        summary = evaluate(
            domain=args.domain,
            model=args.task_model,
            instruction=opt_instruction,
            tasks=dev_tasks,
            max_turns=args.max_turns,
            workers=args.workers,
        )
        print(f"  Dev accuracy: {summary['accuracy_pct']:.1f}% | Avg reward: {summary['avg_reward']:.3f}")
        save_data["dev_eval"] = {k: v for k, v in summary.items() if k != "episodes"}
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)

    return save_data


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_baseline(args):
    """Run baseline (no DSPy optimization)."""
    print(f"=== Tau2 Baseline: domain={args.domain}, model={args.task_model} ===")
    summary = evaluate(
        domain=args.domain,
        model=args.task_model,
        instruction=None,  # uses DEFAULT_INSTRUCTION
        n_eval=args.n_eval,
        split="test" if args.test else "train",
        max_turns=args.max_turns,
        workers=args.workers,
        num_trials=args.num_trials,
    )
    _print_summary(summary)
    _save_results(summary, args.domain, "baseline")
    return summary


def cmd_eval(args):
    """Evaluate saved optimized instruction."""
    instruction = None
    if args.prompt_path:
        data = json.load(open(args.prompt_path))
        instruction = data.get("optimized_instruction", "")
        print(f"Loaded instruction from {args.prompt_path} ({len(instruction)} chars)")
    if not instruction:
        print("No instruction found, using default baseline")

    summary = evaluate(
        domain=args.domain,
        model=args.task_model,
        instruction=instruction,
        n_eval=args.n_eval,
        split="test" if args.test else "train",
        max_turns=args.max_turns,
        workers=args.workers,
        num_trials=args.num_trials,
    )
    _print_summary(summary)
    _save_results(summary, args.domain, "eval")
    return summary


def _print_summary(summary):
    print(f"\n{'='*60}")
    print(f"  Domain:    {summary['domain']}")
    print(f"  Model:     {summary['model']}")
    print(f"  Tasks:     {summary['num_tasks']}")
    print(f"  Trials:    {summary.get('num_trials', 1)} per task ({summary.get('total_trials', summary['num_completed'])} total)")
    print(f"  Pass^1:    {summary.get('binary_pass_rate', 0)*100:.1f}%")
    pass_curve = summary.get("pass_curve", {})
    if pass_curve:
        for k, v in pass_curve.items():
            print(f"  {k}:    {v*100:.1f}%")
    print(f"  Avg Reward:{summary['avg_reward']:.3f}")
    print(f"  Avg Turns: {summary['avg_turns']:.1f}")
    print(f"  Avg Tokens:{summary['avg_tokens']:.0f}")
    print(f"{'='*60}")

    # Per-task breakdown
    print(f"\n  Per-task results:")
    for r in summary["episodes"][:30]:
        if "error" in r:
            print(f"    [ERR]  {r['task_id']}: {r['error'][:60]}")
        else:
            status = "PASS" if r.get("binary_pass") else "FAIL"
            trial_str = f" t{r.get('trial', 0)}" if summary.get("num_trials", 1) > 1 else ""
            print(f"    [{status}] {r['task_id']}{trial_str}: rew={r['reward']:.3f} turns={r.get('turns','?')} "
                  f"term={r.get('termination_reason','?')[:20]}")


def _save_results(summary, domain, prefix):
    out_dir = os.path.join(_REPO, "gen_prompts", "dspy_tau2")
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    out_path = os.path.join(out_dir, f"{prefix}_{domain}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DSPy baseline for tau2-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--domain", required=True, choices=["airline", "retail", "telecom", "mock"])
        p.add_argument("--task-model", default="qwen/qwen-2.5-7b-instruct",
                       help="Model ID for the agent (OpenRouter format, no 'openrouter/' prefix)")
        p.add_argument("--max-turns", type=int, default=12)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--num-trials", type=int, default=1,
                       help="Independent trials per task for pass^k (paper uses k=4)")

    # baseline
    p = subparsers.add_parser("baseline", help="Raw model baseline")
    add_common(p)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--test", action="store_true")

    # train
    p = subparsers.add_parser("train", help="Optimize via MIPROv2")
    add_common(p)
    p.add_argument("--prompt-model", default="openai/gpt-4o-mini")
    p.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-dev", type=int, default=10)

    # eval
    p = subparsers.add_parser("eval", help="Evaluate saved instruction")
    add_common(p)
    p.add_argument("--prompt-path", required=True)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.command == "baseline":
        cmd_baseline(args)
    elif args.command == "train":
        train_dspy(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
