"""
SFT Post-Training for Tau2-bench Policies

Refines RL-trained structure+prompt policies by training on successful (or near-successful)
episodes from tau2 training logs.

Key differences from sft_posttrain.py (standard datasets):
  - Filters by shaped reward (positive = good config) instead of binary tau2 paper metric
  - Includes partial successes (positive shaped reward) since full passes are rare (~3-4%)
  - Reverse-maps tau2 domain-specific tool lists to structure action bitmask indices
  - Reads atom counts from the loaded prompt library (not hardcoded)
  - Always uses API mode (tau2 training is API-based)

Usage:
    # SFT on retail training log using all episodes with tau2 reward > 0
    python sft_posttrain_tau2.py \
        --domain retail \
        --rl-log logs/training_log_qwen-qwen-2.5-7b-instruct_ppo_tau2_retail_1778025410.json \
        --structure-model models/ppo/tau2_retail/qwen-2_5-7b-instruct/structure_policy_tau2_retail_1778041097_ep2750.pt \
        --prompt-model models/ppo/tau2_retail/qwen-2_5-7b-instruct/prompt_policy_tau2_retail_1778041097_ep2750.pt \
        --api-model qwen/qwen-2.5-7b-instruct --epochs 5

    # SFT on telecom with stricter threshold
    python sft_posttrain_tau2.py \
        --domain telecom \
        --rl-log logs/training_log_qwen-qwen-2.5-7b-instruct_ppo_tau2_telecom_1778026917.json \
        --structure-model models/ppo/tau2_telecom/qwen-2_5-7b-instruct/structure_policy_tau2_telecom_1778042005_ep1250.pt \
        --prompt-model models/ppo/tau2_telecom/qwen-2_5-7b-instruct/prompt_policy_tau2_telecom_1778042005_ep1250.pt \
        --api-model qwen/qwen-2.5-7b-instruct --min-reward 0.5 --epochs 5
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("CO_QUIET", "1")

_REPO = os.path.abspath(os.path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO, ".env"))

import logging
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
except Exception:
    pass
for noisy in ("LiteLLM", "litellm", "httpx", "openai", "urllib3", "tau2"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

from configs import load_config
from algorithms.ppo import MultiDiscretePolicyPPO, PolicyNetworkPPO
from algorithms.base import MultiDiscretePolicyGRPO, PolicyNetworkGRPO
from env.structure_env import StructureEnv
from env.prompt_env import PromptEnv
from prompts import library
from tau2_code.src.tool_registry import Tau2ToolRegistry


# ─── Workflow mapping (same as original) ────────────────────────────────────
WORKFLOW_NAMES = {
    "Direct": 0, "Reason+Ans": 1, "Reason+Verify+Ans": 2,
    "Routing": 3, "Parallel-Sectioning": 4, "Parallel-Voting": 5,
    "Orchestrator-Workers": 6, "Evaluator-Optimizer": 7, "Autonomous-Agent": 8,
}

BUDGET_MAP = {"Low": 0, "Mid": 1, "High": 2, "N/A": 0}


def encode_tools_from_names(tool_names: list, tool_registry: Tau2ToolRegistry) -> int:
    """Reverse-map a list of tool names to the group bitmask index.

    For each group, if ANY tool in that group appears in tool_names, set that bit.
    """
    if not tool_names:
        return 0
    wanted = set(tool_names)
    mask = 0
    for i, gname in enumerate(tool_registry.list_groups()):
        group_tools = set(tool_registry.tools_in_group(gname))
        if wanted & group_tools:
            mask |= (1 << i)
    return mask


# ─── Episode loading ────────────────────────────────────────────────────────

def load_tau2_episodes(log_path: str, min_reward: float = 0.0):
    """Load episodes from a tau2 training log, filtered by shaped reward.

    Uses the 'reward' field (shaped/scaled reward) rather than tau2_original_reward
    because shaped reward captures partial-credit signal: correct tool usage,
    fewer errors, clean termination — all things the policy should learn from.
    tau2_original_reward is binary (0 or 1) and discards near-successes.

    Returns (episodes, metadata) where metadata is the top-level log dict minus episodes.
    """
    print(f"Loading log: {log_path}")
    with open(log_path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        all_episodes = raw
        metadata = {}
    else:
        all_episodes = raw.get("episodes", raw.get("data", []))
        metadata = {k: v for k, v in raw.items() if k != "episodes"}

    selected = []
    for ep in all_episodes:
        shaped_r = float(ep.get("reward", ep.get("scaled_reward", 0.0)) or 0.0)
        if shaped_r > min_reward:
            selected.append(ep)

    selected.sort(key=lambda e: float(e.get("reward", e.get("scaled_reward", 0)) or 0), reverse=True)

    print(f"  Total episodes: {len(all_episodes)}")
    print(f"  Selected (shaped_reward > {min_reward}): {len(selected)}")
    if selected:
        shaped = [float(e.get("reward", 0) or 0) for e in selected]
        orig = [float(e.get("tau2_original_reward", 0) or 0) for e in selected]
        print(f"  Shaped reward range: [{min(shaped):.3f}, {max(shaped):.3f}]")
        n_pass = sum(1 for r in orig if r >= 0.999)
        print(f"  Full passes (tau2_original=1): {n_pass}")
        print(f"  Partial successes (shaped>0, tau2_original<1): {len(selected) - n_pass}")
    return selected, metadata


# ─── Structure policy SFT ───────────────────────────────────────────────────

def train_structure_sft(
    policy, episodes, structure_env, tool_registry,
    epochs=5, lr=1e-4, device="cuda", entropy_coef=0.01,
):
    print(f"\n=== Structure Policy SFT ({len(episodes)} episodes, {epochs} epochs) ===")

    from collections import Counter
    wf_dist = Counter(ep.get("workflow", "?") for ep in episodes)
    print("  Workflow distribution:")
    for wf, cnt in wf_dist.most_common():
        print(f"    {wf}: {cnt} ({cnt/len(episodes)*100:.1f}%)")

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        shuffled = episodes.copy()
        random.shuffle(shuffled)

        for ep in tqdm(shuffled, desc=f"struct ep{epoch+1}/{epochs}"):
            question = ep.get("question", "")
            if not question:
                continue

            structure_env.current_q = question
            structure_env.current_a = ""
            structure_env.question_embedding = structure_env.worker.get_embedding(question)
            obs = structure_env._get_observation()

            wf_idx = WORKFLOW_NAMES.get(ep.get("workflow", "Direct"), 0)
            a1_tools = encode_tools_from_names(ep.get("agent1_tools", []), tool_registry)
            a1_budget = BUDGET_MAP.get(ep.get("reasoner_budget", "Mid"), 1)
            a2_tools = encode_tools_from_names(ep.get("agent2_tools", []), tool_registry)
            a2_budget = BUDGET_MAP.get(ep.get("verifier_budget", "Mid"), 1)
            ans_budget = BUDGET_MAP.get(ep.get("answerer_budget", "Mid"), 1)

            target = np.array([wf_idx, a1_tools, a1_budget, a2_tools, a2_budget, ans_budget])
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

            out = policy(obs_t)
            logits_list = out[0] if isinstance(out, tuple) else out

            loss = torch.tensor(0.0, device=device)
            total_ent = torch.tensor(0.0, device=device)
            for i, logits in enumerate(logits_list):
                tgt = torch.LongTensor([target[i]]).to(device)
                if target[i] >= logits.shape[1]:
                    continue
                loss = loss + criterion(logits, tgt)
                probs = torch.softmax(logits, dim=-1)
                total_ent = total_ent - (probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

            loss = loss / len(logits_list)
            ent_loss = total_ent / len(logits_list)
            combined = loss + entropy_coef * ent_loss

            optimizer.zero_grad()
            combined.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / max(n, 1)
        print(f"  Epoch {epoch+1}: loss={avg:.4f} ({n} updates)")

    policy.eval()


# ─── Prompt policy SFT ──────────────────────────────────────────────────────

def train_prompt_sft(
    policy, episodes, prompt_env, tool_registry,
    epochs=5, lr=5e-5, device="cuda",
):
    action_dim = None
    if hasattr(policy, "action_head"):
        action_dim = policy.action_head.out_features
    print(f"\n=== Prompt Policy SFT ({len(episodes)} episodes, {epochs} epochs, action_dim={action_dim}) ===")

    atom_counts = {
        "reasoner": library.NUM_ATOMS.get("reasoner", 6),
        "verifier": library.NUM_ATOMS.get("verifier", 5),
        "answerer": library.NUM_ATOMS.get("answerer", 4),
    }
    print(f"  Atom counts: {atom_counts}")

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6,
    )
    policy.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        skipped = 0
        shuffled = episodes.copy()
        random.shuffle(shuffled)

        for ep in tqdm(shuffled, desc=f"prompt ep{epoch+1}/{epochs}"):
            question = ep.get("question", "")
            if not question:
                continue

            wf_idx = WORKFLOW_NAMES.get(ep.get("workflow", "Direct"), 0)
            a1_tools = encode_tools_from_names(ep.get("agent1_tools", []), tool_registry)
            a1_budget = BUDGET_MAP.get(ep.get("reasoner_budget", "Mid"), 1)
            a2_tools = encode_tools_from_names(ep.get("agent2_tools", []), tool_registry)
            a2_budget = BUDGET_MAP.get(ep.get("verifier_budget", "Mid"), 1)
            ans_budget = BUDGET_MAP.get(ep.get("answerer_budget", "Mid"), 1)

            prompt_env.set_structure(
                question=question,
                answer="",
                embedding=prompt_env.worker.get_embedding(question),
                structure={
                    "workflow_depth": wf_idx,
                    "agent1_tools_idx": a1_tools,
                    "agent1_budget_idx": a1_budget,
                    "agent2_tools_idx": a2_tools,
                    "agent2_budget_idx": a2_budget,
                    "answerer_budget_idx": ans_budget,
                },
            )

            def _train_stage(stage_name, stage_const, prompts_list):
                nonlocal epoch_loss, n, skipped
                if not prompts_list:
                    return
                max_valid = atom_counts.get(stage_name, 4)
                prompt_env.prompt_stage = stage_const
                prompt_env.prompt_step = 0
                prompt_env.selected_prompts[stage_name] = []

                for pidx in prompts_list:
                    if pidx < 1 or (action_dim and pidx >= action_dim):
                        skipped += 1
                        continue

                    obs = prompt_env._get_observation()
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    tgt = torch.LongTensor([pidx]).to(device)

                    out = policy(obs_t)
                    logits = out[0] if isinstance(out, tuple) else out

                    loss = criterion(logits, tgt)
                    if torch.isnan(loss) or torch.isinf(loss):
                        skipped += 1
                        continue

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n += 1

                    prompt_env.selected_prompts[stage_name].append(pidx)
                    prompt_env.prompt_step += 1

            # Reasoner: workflows 1,2,3,4,6,7,8 (not Direct=0, not Parallel-Voting=5)
            if wf_idx not in (0, 5):
                _train_stage("reasoner", prompt_env.PROMPT_STAGE_REASONER,
                             ep.get("reasoner_prompts", []))

            # Verifier: workflows 2, 7
            if wf_idx in (2, 7):
                _train_stage("verifier", prompt_env.PROMPT_STAGE_VERIFIER,
                             ep.get("verifier_prompts", []))

            # Answerer: all workflows
            _train_stage("answerer", prompt_env.PROMPT_STAGE_ANSWERER,
                         ep.get("answerer_prompts", []))

        avg = epoch_loss / max(n, 1)
        scheduler.step(avg)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}: loss={avg:.4f} ({n} steps, {skipped} skipped, lr={cur_lr:.2e})")

    policy.eval()


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SFT post-training for tau2 policies")
    p.add_argument("--domain", required=True, choices=["retail", "telecom", "airline"])
    p.add_argument("--rl-log", required=True, help="Path to tau2 RL training log JSON")
    p.add_argument("--structure-model", required=True, help="Structure policy checkpoint")
    p.add_argument("--prompt-model", required=True, help="Prompt policy checkpoint")
    p.add_argument("--api-model", required=True, help="OpenRouter model id (e.g. qwen/qwen-2.5-7b-instruct)")
    p.add_argument("--config", default="hierarchical")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--struct-lr", type=float, default=1e-4)
    p.add_argument("--prompt-lr", type=float, default=5e-5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--min-reward", type=float, default=0.0,
                   help="Minimum shaped reward threshold (default: 0.0 = include all positive-reward episodes)")
    p.add_argument("--output-dir", default="models/sft_posttrained")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_name = f"tau2_{args.domain}"

    # ── Load config + atoms ──
    cfg = load_config(args.config)
    cfg.DATASET_NAME = dataset_name

    atoms_path = library._get_atoms_path(dataset_name)
    if os.path.exists(atoms_path):
        library.load_or_create_atoms(dataset_name, worker=None)
    else:
        print(f"  WARNING: No atoms at {atoms_path}; using base atoms only.")
    print(f"  Atom counts: {library.NUM_ATOMS}")

    # ── Load episodes ──
    episodes, meta = load_tau2_episodes(args.rl_log, min_reward=args.min_reward)
    if not episodes:
        print("ERROR: No episodes passed the reward filter. Try lowering --min-reward.")
        return

    # ── Load checkpoints ──
    struct_ckpt = torch.load(args.structure_model, map_location=device, weights_only=False)
    prompt_ckpt = torch.load(args.prompt_model, map_location=device, weights_only=False)

    has_value = any("value_head" in k for k in struct_ckpt.get("model_state_dict", struct_ckpt).keys())
    algorithm = "ppo" if has_value else "grpo"
    print(f"  Detected algorithm: {algorithm.upper()}")

    if algorithm == "ppo":
        struct_policy = MultiDiscretePolicyPPO(
            obs_dim=struct_ckpt["obs_dim"], action_dims=struct_ckpt["action_dims"],
        ).to(device)
        prompt_policy = PolicyNetworkPPO(
            obs_dim=prompt_ckpt["obs_dim"], action_dim=prompt_ckpt["action_dim"],
        ).to(device)
    else:
        struct_policy = MultiDiscretePolicyGRPO(
            obs_dim=struct_ckpt["obs_dim"], action_dims=struct_ckpt["action_dims"],
        ).to(device)
        prompt_policy = PolicyNetworkGRPO(
            obs_dim=prompt_ckpt["obs_dim"], action_dim=prompt_ckpt["action_dim"],
        ).to(device)

    struct_policy.load_state_dict(struct_ckpt["model_state_dict"])
    prompt_policy.load_state_dict(prompt_ckpt["model_state_dict"])
    print(f"  Loaded structure ({struct_ckpt['obs_dim']}→{struct_ckpt['action_dims']}) + "
          f"prompt ({prompt_ckpt['obs_dim']}→{prompt_ckpt['action_dim']})")

    # ── Create envs (for observation generation only) ──
    tool_registry = Tau2ToolRegistry(args.domain)
    structure_env = StructureEnv(cfg, use_api=True, api_model=args.api_model)
    prompt_env = PromptEnv(cfg, use_api=True, api_model=args.api_model)

    # ── Train ──
    train_structure_sft(
        struct_policy, episodes, structure_env, tool_registry,
        epochs=args.epochs, lr=args.struct_lr, device=device,
        entropy_coef=args.entropy_coef,
    )
    train_prompt_sft(
        prompt_policy, episodes, prompt_env, tool_registry,
        epochs=args.epochs, lr=args.prompt_lr, device=device,
    )

    # ── Save ──
    ts = int(time.time())
    model_slug = args.api_model.replace("/", "-")
    save_dir = os.path.join(args.output_dir, dataset_name, model_slug)
    os.makedirs(save_dir, exist_ok=True)

    s_path = os.path.join(save_dir, f"structure_policy_sft_{ts}.pt")
    p_path = os.path.join(save_dir, f"prompt_policy_sft_{ts}.pt")

    torch.save({
        "model_state_dict": struct_policy.state_dict(),
        "action_dims": struct_ckpt["action_dims"],
        "obs_dim": struct_ckpt["obs_dim"],
        "algorithm": f"{algorithm.upper()}_SFT",
        "dataset": dataset_name,
        "base_model": args.api_model,
        "source_rl_checkpoint": args.structure_model,
        "min_reward_filter": args.min_reward,
        "n_episodes": len(episodes),
    }, s_path)

    torch.save({
        "model_state_dict": prompt_policy.state_dict(),
        "action_dim": prompt_ckpt["action_dim"],
        "obs_dim": prompt_ckpt["obs_dim"],
        "algorithm": f"{algorithm.upper()}_SFT",
        "dataset": dataset_name,
        "base_model": args.api_model,
        "source_rl_checkpoint": args.prompt_model,
        "min_reward_filter": args.min_reward,
        "n_episodes": len(episodes),
    }, p_path)

    # ── Log ──
    log_dir = os.path.join("logs", "sft_train", dataset_name, model_slug)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"sft_tau2_{ts}.json")
    with open(log_path, "w") as f:
        json.dump({
            "domain": args.domain,
            "dataset": dataset_name,
            "algorithm": algorithm,
            "api_model": args.api_model,
            "source_rl_log": args.rl_log,
            "source_structure_ckpt": args.structure_model,
            "source_prompt_ckpt": args.prompt_model,
            "min_reward_filter": args.min_reward,
            "n_episodes": len(episodes),
            "epochs": args.epochs,
            "struct_lr": args.struct_lr,
            "prompt_lr": args.prompt_lr,
            "entropy_coef": args.entropy_coef,
            "output_structure": s_path,
            "output_prompt": p_path,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SFT complete!")
    print(f"{'='*60}")
    print(f"  Episodes used: {len(episodes)}")
    print(f"  Structure: {s_path}")
    print(f"  Prompt:    {p_path}")
    print(f"  Log:       {log_path}")
    print(f"\nEvaluate with:")
    print(f"  python tau2_code/scripts/eval_tau2.py --domain {args.domain} --agent-type hrl \\")
    print(f"    --structure-model {s_path} --prompt-model {p_path} \\")
    print(f"    --api-model {args.api_model} --episodes all --workers 10 --num-trials 5")


if __name__ == "__main__":
    main()
