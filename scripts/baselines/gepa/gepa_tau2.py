"""GEPA baseline for tau2-bench.

GEPA (Genetic Evolutionary Prompt Adaptation) uses reflective evolution to optimize
the agent's system instruction. Unlike plain DSPy/MIPROv2 which tries random variations,
GEPA analyzes failure traces with a strong reflection LLM and proposes improved
instructions iteratively.

Usage:
    # Baseline (no optimization, raw model)
    python scripts/baselines/gepa/gepa_tau2.py baseline --domain telecom --n-eval 10

    # Train (optimize via GEPA evolutionary reflection)
    python scripts/baselines/gepa/gepa_tau2.py train --domain telecom --n-train 20 --n-dev 10

    # Evaluate saved optimized instruction
    python scripts/baselines/gepa/gepa_tau2.py eval --domain telecom \
        --prompt-path gen_prompts/gepa_tau2/telecom_heavy.json --n-eval 50
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO, ".env"))

os.environ.setdefault("LITELLM_LOG", "ERROR")

import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False

import dspy
from dspy.teleprompt import GEPA


# ---------------------------------------------------------------------------
# Tool descriptions
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
# Simple LLM agent
# ---------------------------------------------------------------------------

class SimpleLLMAgent:
    """Minimal agent: system prompt + conversation -> next action via litellm."""

    def __init__(self, model: str, system_prompt: str, max_tokens: int = 1024):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages = [{"role": "system", "content": system_prompt}]
        self.total_tokens = 0

    def respond(self, observation: str) -> str:
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
# Tau2 episode runner
# ---------------------------------------------------------------------------

_TOOL_RE = re.compile(r"TOOL:\s*([A-Za-z_]\w*)\s*\|\|\s*QUERY:\s*(.+)", re.DOTALL)


def _action_to_tau2(action_text: str) -> str:
    """Convert 'TOOL: name || QUERY: {args}' to tau2 JSON format."""
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
    """Run one tau2 episode. Returns result dict with reward, transcript, etc."""
    import gymnasium as gym
    from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID

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
    turn = 0

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

    from tau2_code.src.rollout import _parse_reward_info
    parsed = _parse_reward_info(info or {})

    return {
        "task_id": task_id,
        "reward": float(reward) if reward else 0.0,
        "turns": turn + 1,
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


# ---------------------------------------------------------------------------
# System prompt builder
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
# Dataset loading
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    domain: str,
    model: str,
    instruction: str = None,
    tasks: List[Dict] = None,
    n_eval: Optional[int] = None,
    split: str = "train",
    max_turns: int = 12,
    workers: int = 4,
) -> Dict:
    """Run evaluation on tau2 tasks."""
    from tqdm import tqdm

    if tasks is None:
        tasks = load_tasks(domain, split=split, n=n_eval)

    system_prompt = build_system_prompt(domain, instruction)
    results = []

    def run_one(task):
        return run_episode(
            domain=domain,
            task_id=task["task_id"],
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
        )

    if workers <= 1:
        for task in tqdm(tasks, desc="Evaluating"):
            try:
                results.append(run_one(task))
            except Exception as e:
                print(f"  Error on {task['task_id']}: {e}")
                results.append({"task_id": task["task_id"], "reward": 0.0, "error": str(e)})
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, t): t for t in tasks}
            with tqdm(total=len(tasks), desc="Evaluating") as pbar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"  Error on {task['task_id']}: {e}")
                        results.append({"task_id": task["task_id"], "reward": 0.0, "error": str(e)})
                    pbar.update(1)
                    valid = [r for r in results if "error" not in r]
                    if valid:
                        avg_r = sum(r["reward"] for r in valid) / len(valid)
                        corr = sum(1 for r in valid if r["reward"] > 0.5)
                        pbar.set_postfix(avg_rew=f"{avg_r:.3f}", correct=f"{corr}/{len(valid)}")

    valid = [r for r in results if "error" not in r]
    avg_reward = sum(r["reward"] for r in valid) / len(valid) if valid else 0.0
    correct = sum(1 for r in valid if r["reward"] > 0.5)

    return {
        "domain": domain,
        "model": model,
        "instruction": (instruction or DEFAULT_INSTRUCTION)[:500],
        "split": split,
        "num_tasks": len(tasks),
        "num_completed": len(valid),
        "avg_reward": avg_reward,
        "correct": correct,
        "accuracy_pct": correct / len(valid) * 100 if valid else 0.0,
        "avg_turns": sum(r.get("turns", 0) for r in valid) / len(valid) if valid else 0,
        "avg_tokens": sum(r.get("total_tokens", 0) for r in valid) / len(valid) if valid else 0,
        "timestamp": datetime.now().isoformat(),
        "episodes": results,
    }


# ---------------------------------------------------------------------------
# GEPA Training (evolutionary reflection optimization)
# ---------------------------------------------------------------------------

class Tau2AgentSignature(dspy.Signature):
    """You are optimizing instructions for a customer-support agent that handles
    multi-turn dialogs with tool access. The agent needs to identify customers,
    diagnose issues, use tools to resolve them, and confirm resolution."""
    task_context: str = dspy.InputField(desc="Customer support scenario and domain context")
    agent_prompt: str = dspy.OutputField(desc="Optimized system instructions for the agent")


class Tau2Solver(dspy.Module):
    """DSPy module wrapping the tau2 agent. GEPA optimizes its instruction."""

    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(Tau2AgentSignature)

    def forward(self, task_context: str) -> dspy.Prediction:
        return self.cot(task_context=task_context)


def train_gepa(args):
    """Optimize system instruction via GEPA evolutionary reflection."""
    print(f"=== GEPA Tau2 Train: domain={args.domain}, model={args.task_model} ===")

    # Load tasks
    tasks = load_tasks(args.domain, split="train", n=args.n_train + args.n_dev)
    train_tasks = tasks[:args.n_train]
    dev_tasks = tasks[args.n_train:args.n_train + args.n_dev]
    print(f"  Train: {len(train_tasks)}, Dev: {len(dev_tasks)}")

    # Build domain context for DSPy examples
    tool_names = get_tool_names(args.domain)
    domain_context = (
        f"Domain: {args.domain} customer support\n"
        f"Available tools: {', '.join(tool_names)}\n\n"
        f"Example scenarios:\n"
    )
    for t in train_tasks[:5]:
        domain_context += f"- {t['description'][:150]}\n"

    # Create DSPy train/dev examples
    train_examples = [
        dspy.Example(task_context=domain_context, task_id=t["task_id"]).with_inputs("task_context")
        for t in train_tasks
    ]
    dev_examples = [
        dspy.Example(task_context=domain_context, task_id=t["task_id"]).with_inputs("task_context")
        for t in dev_tasks
    ]

    # Metric function: GEPA requires 5 args (gold, pred, trace, pred_name, pred_trace)
    def metric_fn(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
        instr = getattr(prediction, "agent_prompt", "") or ""
        if not instr or len(instr) < 20:
            return 0.0

        task_id = getattr(example, "task_id", "")
        if not task_id:
            import random
            task_id = random.choice(dev_tasks)["task_id"]

        try:
            result = run_episode(
                domain=args.domain,
                task_id=task_id,
                system_prompt=build_system_prompt(args.domain, instr),
                model=args.task_model,
                max_turns=args.max_turns,
            )
            reward = max(0.0, min(1.0, result["reward"]))
            return reward
        except Exception as e:
            print(f"  [metric error] {e}")
            return 0.0

    # Configure DSPy LMs
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    task_model_name = args.task_model
    if not task_model_name.startswith("openrouter/"):
        task_model_name = f"openrouter/{task_model_name}"

    task_lm = dspy.LM(
        task_model_name,
        api_key=openrouter_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=4096,
        temperature=0.0,
    )

    # Reflection LM (stronger model for analyzing failures)
    reflection_model = args.reflection_model
    if not reflection_model.startswith("openrouter/"):
        reflection_model = f"openrouter/{reflection_model}"

    reflection_lm = dspy.LM(
        reflection_model,
        api_key=openrouter_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=8000,
        temperature=1.0,
    )

    dspy.configure(lm=task_lm)

    # Create solver
    solver = Tau2Solver()

    # Set up log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(_REPO) / "scripts" / "baselines" / "gepa" / "gepa_logs" / "tau2" / args.domain / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Reflection model: {reflection_model}")
    print(f"  Optimization mode: {args.auto}")
    print(f"  Log dir: {log_dir}")
    print(f"\n  Running GEPA optimization...")

    optimizer = GEPA(
        metric=metric_fn,
        auto=args.auto,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.reflection_batch,
        num_threads=1,
        track_stats=True,
        skip_perfect_score=True,
        log_dir=str(log_dir),
    )

    optimized_solver = optimizer.compile(
        student=solver,
        trainset=train_examples,
        valset=dev_examples,
    )

    # Extract optimized instruction
    opt_instruction = ""
    try:
        state_str = optimized_solver.save()
        state = json.loads(state_str) if isinstance(state_str, str) else state_str
        for key, val in state.items():
            if isinstance(val, dict) and "signature" in val:
                opt_instruction = val["signature"].get("instructions", "")
                if opt_instruction:
                    break
    except Exception as e:
        print(f"  Warning: could not extract from save(): {e}")

    if not opt_instruction:
        try:
            opt_instruction = str(optimized_solver.cot.predict.signature.instructions)
        except Exception:
            pass

    # Try to get from detailed_results
    if not opt_instruction and hasattr(optimized_solver, "detailed_results"):
        try:
            best = optimized_solver.detailed_results.best_candidate
            if best:
                opt_instruction = str(list(best.values())[0]) if isinstance(best, dict) else str(best)
        except Exception:
            pass

    print(f"\n  Optimized instruction ({len(opt_instruction)} chars):")
    print(f"  {opt_instruction[:400]}...")

    # Save results
    out_dir = os.path.join(_REPO, "gen_prompts", "gepa_tau2")
    os.makedirs(out_dir, exist_ok=True)
    model_safe = args.task_model.replace("/", "_")
    out_path = os.path.join(out_dir, f"{args.domain}_{args.auto}_{model_safe}.json")

    save_data = {
        "domain": args.domain,
        "task_model": args.task_model,
        "reflection_model": args.reflection_model,
        "auto": args.auto,
        "n_train": len(train_tasks),
        "n_dev": len(dev_tasks),
        "optimized_instruction": opt_instruction,
        "log_dir": str(log_dir),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        save_data["dspy_state"] = optimized_solver.save()
    except Exception:
        pass

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved to: {out_path}")

    # Post-optimization eval on dev set
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
    """Run raw baseline (no optimization)."""
    print(f"=== GEPA Tau2 Baseline: domain={args.domain}, model={args.task_model} ===")
    summary = evaluate(
        domain=args.domain,
        model=args.task_model,
        instruction=None,
        n_eval=args.n_eval,
        split="test" if args.test else "train",
        max_turns=args.max_turns,
        workers=args.workers,
    )
    _print_summary(summary)
    _save_results(summary, args.domain, "baseline")
    return summary


def cmd_eval(args):
    """Evaluate a saved optimized instruction."""
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
    )
    _print_summary(summary)
    _save_results(summary, args.domain, "eval")
    return summary


def _print_summary(summary):
    print(f"\n{'='*60}")
    print(f"  Domain:    {summary['domain']}")
    print(f"  Model:     {summary['model']}")
    print(f"  Tasks:     {summary['num_completed']}/{summary['num_tasks']}")
    print(f"  Accuracy:  {summary['accuracy_pct']:.1f}% ({summary['correct']}/{summary['num_completed']})")
    print(f"  Avg Reward:{summary['avg_reward']:.3f}")
    print(f"  Avg Turns: {summary['avg_turns']:.1f}")
    print(f"  Avg Tokens:{summary['avg_tokens']:.0f}")
    print(f"{'='*60}")

    print(f"\n  Per-task results:")
    for r in summary["episodes"][:30]:
        if "error" in r:
            print(f"    [ERR]  {r['task_id']}: {r['error'][:60]}")
        else:
            status = "PASS" if r.get("reward", 0) > 0.5 else "FAIL"
            print(f"    [{status}] {r['task_id']}: rew={r['reward']:.3f} turns={r.get('turns','?')} "
                  f"term={r.get('termination_reason','?')[:20]}")


def _save_results(summary, domain, prefix):
    out_dir = os.path.join(_REPO, "gen_prompts", "gepa_tau2")
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
    parser = argparse.ArgumentParser(description="GEPA baseline for tau2-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--domain", required=True, choices=["airline", "retail", "telecom", "mock"])
        p.add_argument("--task-model", default="openrouter/qwen/qwen-2.5-7b-instruct",
                       help="LiteLLM model ID for the agent")
        p.add_argument("--max-turns", type=int, default=12)
        p.add_argument("--workers", type=int, default=4)

    # baseline
    p = subparsers.add_parser("baseline", help="Raw model baseline")
    add_common(p)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--test", action="store_true")

    # train
    p = subparsers.add_parser("train", help="Optimize via GEPA reflection")
    add_common(p)
    p.add_argument("--reflection-model", default="openai/gpt-4o-mini",
                   help="Strong LLM for reflection (routes through OpenRouter)")
    p.add_argument("--auto", default="heavy", choices=["light", "medium", "heavy"])
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-dev", type=int, default=10)
    p.add_argument("--reflection-batch", type=int, default=3,
                   help="Examples to reflect on per iteration")

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
        train_gepa(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
