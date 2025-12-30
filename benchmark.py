"""
Benchmark script to compare trained RL controllers.
Supports single_step, multi_step, and hierarchical (dual-policy) environments.

Usage:
    python benchmark.py --config single_step          # Benchmark single-step
    python benchmark.py --config multi_step           # Benchmark multi-step
    python benchmark.py --compare                     # Compare single vs multi-step
    python benchmark.py --compare --episodes 50       # Compare with more episodes
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from configs import load_config
from env import GeneralAgentEnv, MultiStepAgentEnv, StructureEnv, PromptEnv


def get_latest_model(env_mode: str = None, policy_type: str = None):
    """
    Find the latest model in the models directory.
    
    Args:
        env_mode: Filter by environment mode (single_step, multi_step, hierarchical)
        policy_type: For hierarchical, specify 'structure' or 'prompt'
    """
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    if policy_type:
        pattern = os.path.join(models_dir, f"{policy_type}_policy_*.zip")
    elif env_mode:
        pattern = os.path.join(models_dir, f"controller_{env_mode}_*.zip")
    else:
        pattern = os.path.join(models_dir, "controller_*.zip")
    
    list_of_files = glob.glob(pattern)
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return os.path.splitext(latest_file)[0]


def run_episode(env, policy_fn, is_multistep=False):
    """
    Run a single episode and return results.
    """
    obs = env.reset()
    done = False
    total_reward = 0.0
    num_steps = 0
    
    while not done:
        action = policy_fn(obs)
        
        # Handle action format
        if isinstance(action, np.ndarray) and action.ndim == 0:
            action = action.item()
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        obs, rewards, dones, infos = env.step([action] if action.ndim == 0 else action)
        
        total_reward += float(rewards[0])
        num_steps += 1
        done = dones[0]
        
        # For single-step, we're done after one step
        if not is_multistep:
            break
    
    return infos[0], total_reward, num_steps


def run_benchmark_single_step(cfg, model_path: str = None, num_episodes: int = 30):
    """Run benchmark for single-step environment with baseline comparisons."""
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: Single-Step Environment")
    print(f"Episodes per Strategy: {num_episodes}")
    print(f"Dataset: {cfg.DATASET_NAME}")
    print(f"{'='*70}\n")
    
    env = DummyVecEnv([lambda: GeneralAgentEnv(cfg)])
    strategies = {}
    
    # Load RL model if provided
    if model_path:
        real_path = model_path if model_path.endswith('.zip') else model_path + ".zip"
        if os.path.exists(real_path):
            print(f"Loading model: {real_path}")
            rl_model = PPO.load(model_path, device='cpu')
            strategies["RL (Learned)"] = lambda obs: rl_model.predict(obs, deterministic=True)[0][0]
        else:
            print(f"Warning: Model not found at {real_path}")
    
    # Baseline configurations
    baselines = {
        "Direct-Low": [0, 0, 0, 0, 0, 0],
        "Direct-High": [0, 0, 2, 0, 2, 2],
        "Reason-Mid": [1, 0, 1, 0, 1, 1],
        "Reason-Calc-Mid": [1, 1, 1, 0, 1, 1],
        "Full-Mid": [2, 0, 1, 0, 1, 1],
    }
    
    for name, cfg_vals in baselines.items():
        strategies[name] = lambda obs, c=cfg_vals: np.array(c)
    
    strategies["Random"] = lambda obs: env.action_space.sample()
    
    return _run_strategies(env, strategies, num_episodes, cfg, is_multistep=False)


def run_benchmark_multistep(cfg, model_path: str = None, num_episodes: int = 30):
    """Run benchmark for multi-step environment."""
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: Multi-Step Environment")
    print(f"Episodes: {num_episodes}")
    print(f"Dataset: {cfg.DATASET_NAME}")
    print(f"{'='*70}\n")
    
    env = DummyVecEnv([lambda: MultiStepAgentEnv(cfg)])
    strategies = {}
    
    if model_path:
        real_path = model_path if model_path.endswith('.zip') else model_path + ".zip"
        if os.path.exists(real_path):
            print(f"Loading model: {real_path}")
            rl_model = PPO.load(model_path, device='cpu')
            strategies["RL (Learned)"] = lambda obs: rl_model.predict(obs, deterministic=True)[0]
        else:
            print(f"Warning: Model not found at {real_path}")
    
    strategies["Random"] = lambda obs: env.action_space.sample()
    
    if not strategies:
        print("Error: No strategies available.")
        return None, None
    
    return _run_strategies(env, strategies, num_episodes, cfg, is_multistep=True)


def run_benchmark_hierarchical(cfg, structure_model_path: str = None, 
                                prompt_model_path: str = None, num_episodes: int = 30):
    """
    Run benchmark for hierarchical (dual-policy) environment.
    Requires both structure and prompt models.
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: Hierarchical Environment (Dual-Policy)")
    print(f"Episodes: {num_episodes}")
    print(f"Dataset: {cfg.DATASET_NAME}")
    print(f"{'='*70}\n")
    
    results = []
    detailed_results = []
    
    # Strategy definitions
    strategies = {}
    
    # Load RL models if provided
    if structure_model_path and prompt_model_path:
        struct_path = structure_model_path if structure_model_path.endswith('.zip') else structure_model_path + ".zip"
        prompt_path = prompt_model_path if prompt_model_path.endswith('.zip') else prompt_model_path + ".zip"
        
        if os.path.exists(struct_path) and os.path.exists(prompt_path):
            print(f"Loading structure model: {struct_path}")
            print(f"Loading prompt model: {prompt_path}")
            structure_model = PPO.load(structure_model_path, device='cpu')
            prompt_model = PPO.load(prompt_model_path, device='cpu')
            strategies["RL (Learned)"] = ("rl", structure_model, prompt_model)
        else:
            print(f"Warning: Models not found")
    
    # Random baseline
    strategies["Random"] = ("random", None, None)
    
    # Run each strategy
    structure_env = StructureEnv(cfg)
    prompt_env = PromptEnv(cfg)
    
    for name, (strategy_type, struct_model, prompt_model) in strategies.items():
        print(f"Testing: {name}...", end=" ", flush=True)
        start_time = time.time()
        
        accuracies = []
        tokens_list = []
        rewards_list = []
        workflows = []
        decision_steps_list = []
        
        for ep in range(num_episodes):
            # Step 1: Structure decision
            struct_obs, struct_info = structure_env.reset()
            
            if strategy_type == "rl":
                struct_action, _ = struct_model.predict(struct_obs, deterministic=True)
            else:  # random
                struct_action = structure_env.action_space.sample()
            
            _, struct_reward, _, _, struct_exec_info = structure_env.step(struct_action)
            total_reward = struct_reward
            
            # Step 2: Prompt selection
            prompt_env.set_structure(
                question=struct_exec_info["question"],
                answer=struct_exec_info["answer"],
                embedding=struct_exec_info["embedding"],
                structure=struct_exec_info["structure"]
            )
            
            prompt_obs, _ = prompt_env.reset()
            done = False
            prompt_steps = 0
            
            while not done:
                if strategy_type == "rl":
                    prompt_action, _ = prompt_model.predict(prompt_obs, deterministic=True)
                else:  # random
                    prompt_action = prompt_env.action_space.sample()
                
                prompt_obs, reward, done, _, info = prompt_env.step(prompt_action)
                total_reward += reward
                prompt_steps += 1
            
            # Record results
            accuracies.append(1 if info.get("correct", False) else 0)
            tokens_list.append(info.get("total_tokens", 0))
            rewards_list.append(total_reward)
            workflows.append(struct_exec_info["workflow_name"])
            decision_steps_list.append(1 + prompt_steps)  # 1 for structure + prompt steps
            
            detailed_results.append({
                "strategy": name,
                "episode": ep,
                "correct": info.get("correct", False),
                "workflow": struct_exec_info["workflow_name"],
                "llm_steps": info.get("steps_taken", 1),
                "decision_steps": 1 + prompt_steps,
                "tokens": info.get("total_tokens", 0),
                "reward": total_reward,
                "reasoner_prompts": str(info.get("reasoner_prompts", [])),
                "verifier_prompts": str(info.get("verifier_prompts", [])),
                "answerer_prompts": str(info.get("answerer_prompts", [])),
            })
        
        elapsed = time.time() - start_time
        
        avg_acc = np.mean(accuracies)
        avg_tokens = np.mean(tokens_list)
        avg_reward = np.mean(rewards_list)
        avg_decisions = np.mean(decision_steps_list)
        
        results.append({
            "Strategy": name,
            "Accuracy": f"{avg_acc:.1%}",
            "Avg Reward": f"{avg_reward:.3f}",
            "Avg Decisions": f"{avg_decisions:.1f}",
            "Avg Tokens": f"{avg_tokens:.0f}",
            "Time (s)": f"{elapsed:.1f}"
        })
        
        print(f"Accuracy: {avg_acc:.1%}, Reward: {avg_reward:.3f}")
    
    # Print results
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    output_path = f"benchmark_results_{cfg.ENV_MODE}_{cfg.DATASET_NAME}_{int(time.time())}.csv"
    detailed_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return df, detailed_df


def run_comparison_benchmark(num_episodes: int = 30, 
                              single_step_model: str = None,
                              multi_step_model: str = None):
    """
    Compare single_step and multi_step models on the same questions.
    Runs both models on identical questions for fair comparison.
    """
    print(f"\n{'='*70}")
    print(f"COMPARISON BENCHMARK: Single-Step vs Multi-Step")
    print(f"Episodes per strategy: {num_episodes}")
    print(f"{'='*70}\n")
    
    # Load both configs
    cfg_single = load_config("single_step")
    cfg_multi = load_config("multi_step")
    
    # Create environments
    single_env = DummyVecEnv([lambda: GeneralAgentEnv(cfg_single)])
    multi_env = DummyVecEnv([lambda: MultiStepAgentEnv(cfg_multi)])
    
    # Find models if not provided
    if not single_step_model:
        single_step_model = get_latest_model("single_step")
        if not single_step_model:
            # Try to find any controller model (backwards compatibility)
            single_step_model = get_latest_model()
    
    if not multi_step_model:
        multi_step_model = get_latest_model("multi_step")
    
    # Load models
    models = {}
    
    if single_step_model:
        path = single_step_model + ".zip" if not single_step_model.endswith('.zip') else single_step_model
        if os.path.exists(path):
            print(f"Loading single-step model: {path}")
            models["Single-Step RL"] = ("single", PPO.load(single_step_model, device='cpu'))
        else:
            print(f"Warning: Single-step model not found at {path}")
    
    if multi_step_model:
        path = multi_step_model + ".zip" if not multi_step_model.endswith('.zip') else multi_step_model
        if os.path.exists(path):
            print(f"Loading multi-step model: {path}")
            models["Multi-Step RL"] = ("multi", PPO.load(multi_step_model, device='cpu'))
        else:
            print(f"Warning: Multi-step model not found at {path}")
    
    # Add random baselines
    models["Random (Single-Step)"] = ("single", None)
    models["Random (Multi-Step)"] = ("multi", None)
    
    if len(models) == 2:  # Only random baselines
        print("\nWarning: No trained models found. Only running random baselines.")
    
    # Run comparison
    results = []
    detailed_results = []
    
    for name, (mode, model) in models.items():
        print(f"\nTesting: {name}...", end=" ", flush=True)
        start_time = time.time()
        
        env = single_env if mode == "single" else multi_env
        is_multistep = (mode == "multi")
        
        accuracies = []
        tokens_list = []
        rewards_list = []
        workflows = []
        steps_list = []
        decision_steps_list = []
        
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            decision_steps = 0
            
            while not done:
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                
                # Ensure action is properly formatted for vectorized env
                if not isinstance(action, np.ndarray):
                    action = np.array(action)
                
                # DummyVecEnv expects batch of actions: [action] for single env
                # For MultiDiscrete (1D array), we need to add batch dimension
                if action.ndim == 0:
                    # Scalar action (Discrete)
                    action_batch = np.array([action.item()])
                elif action.ndim == 1:
                    # 1D action (MultiDiscrete) - add batch dimension
                    action_batch = action.reshape(1, -1)
                else:
                    # Already batched
                    action_batch = action
                
                obs, rewards, dones, infos = env.step(action_batch)
                total_reward += float(rewards[0])
                decision_steps += 1
                done = dones[0]
                
                if not is_multistep:
                    break
            
            info = infos[0]
            accuracies.append(1 if info["correct"] else 0)
            tokens_list.append(info["total_tokens"])
            rewards_list.append(total_reward)
            workflows.append(info["workflow"])
            steps_list.append(info["steps_taken"])
            decision_steps_list.append(decision_steps)
            
            detailed_results.append({
                "strategy": name,
                "mode": mode,
                "episode": ep,
                "correct": info["correct"],
                "workflow": info["workflow"],
                "llm_steps": info["steps_taken"],
                "decision_steps": decision_steps,
                "tokens": info["total_tokens"],
                "reward": total_reward,
            })
        
        elapsed = time.time() - start_time
        avg_acc = np.mean(accuracies)
        avg_tokens = np.mean(tokens_list)
        avg_reward = np.mean(rewards_list)
        avg_steps = np.mean(steps_list)
        avg_decisions = np.mean(decision_steps_list)
        
        results.append({
            "Strategy": name,
            "Mode": mode,
            "Accuracy": f"{avg_acc:.1%}",
            "Avg Reward": f"{avg_reward:.3f}",
            "Avg LLM Steps": f"{avg_steps:.2f}",
            "Avg Decisions": f"{avg_decisions:.1f}",
            "Avg Tokens": f"{avg_tokens:.0f}",
            "Time (s)": f"{elapsed:.1f}"
        })
        
        print(f"Accuracy: {avg_acc:.1%}, Reward: {avg_reward:.3f}")
    
    # Print results
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("COMPARISON RESULTS: Single-Step vs Multi-Step")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    # Workflow distribution
    detailed_df = pd.DataFrame(detailed_results)
    print("\nWorkflow Distribution by Strategy:")
    print(detailed_df.groupby(["strategy", "workflow"]).size().unstack(fill_value=0))
    
    # Save
    output_path = f"benchmark_comparison_{int(time.time())}.csv"
    detailed_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return df, detailed_df


def _run_strategies(env, strategies, num_episodes, cfg, is_multistep=False):
    """Run all strategies and collect results."""
    results = []
    detailed_results = []
    
    for name, policy_fn in strategies.items():
        print(f"Testing: {name}...", end=" ", flush=True)
        start_time = time.time()
        
        accuracies = []
        steps_list = []
        tools_list = []
        tokens_list = []
        workflows = []
        rewards_list = []
        decision_steps_list = []
        
        for ep in range(num_episodes):
            info, total_reward, num_steps = run_episode(env, policy_fn, is_multistep)
            
            accuracies.append(1 if info["correct"] else 0)
            steps_list.append(info["steps_taken"])
            tools_list.append(info["tools_used"])
            tokens_list.append(info["total_tokens"])
            workflows.append(info["workflow"])
            rewards_list.append(total_reward)
            decision_steps_list.append(num_steps)
            
            detailed_results.append({
                "strategy": name,
                "episode": ep,
                "correct": info["correct"],
                "workflow": info["workflow"],
                "llm_steps": info["steps_taken"],
                "decision_steps": num_steps,
                "tools": info["tools_used"],
                "tokens": info["total_tokens"],
                "reward": total_reward,
            })
        
        elapsed = time.time() - start_time
        
        avg_acc = np.mean(accuracies)
        avg_steps = np.mean(steps_list)
        avg_decision_steps = np.mean(decision_steps_list)
        avg_tools = np.mean(tools_list)
        avg_tokens = np.mean(tokens_list)
        avg_reward = np.mean(rewards_list)
        
        results.append({
            "Strategy": name,
            "Accuracy": f"{avg_acc:.1%}",
            "Avg Reward": f"{avg_reward:.3f}",
            "Avg LLM Steps": f"{avg_steps:.2f}",
            "Avg Decisions": f"{avg_decision_steps:.1f}",
            "Avg Tools": f"{avg_tools:.2f}",
            "Avg Tokens": f"{avg_tokens:.0f}",
            "Time (s)": f"{elapsed:.1f}"
        })
        
        print(f"Accuracy: {avg_acc:.1%}, Reward: {avg_reward:.3f}")
    
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    detailed_df = pd.DataFrame(detailed_results)
    output_path = f"benchmark_results_{cfg.ENV_MODE}_{cfg.DATASET_NAME}_{int(time.time())}.csv"
    detailed_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return df, detailed_df


def analyze_results(detailed_df: pd.DataFrame):
    """Analyze benchmark results."""
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    agg_cols = {
        "correct": ["mean", "std", "sum"],
        "tokens": ["mean", "std"],
        "reward": ["mean", "std"]
    }
    
    summary = detailed_df.groupby("strategy").agg(agg_cols).round(3)
    print("\nPer-Strategy Statistics:")
    print(summary)
    
    print("\nWorkflow Distribution:")
    workflow_dist = detailed_df.groupby(["strategy", "workflow"]).size().unstack(fill_value=0)
    print(workflow_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RL Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hierarchical",
        choices=["single_step", "multi_step", "hierarchical"],
        help="Configuration/environment to benchmark"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare single_step and multi_step models against each other"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (for single_step/multi_step)"
    )
    parser.add_argument(
        "--single-step-model",
        type=str,
        default=None,
        help="Path to single-step model (for --compare mode)"
    )
    parser.add_argument(
        "--multi-step-model",
        type=str,
        default=None,
        help="Path to multi-step model (for --compare mode)"
    )
    parser.add_argument(
        "--structure-model",
        type=str,
        default=None,
        help="Path to structure policy model (for hierarchical)"
    )
    parser.add_argument(
        "--prompt-model",
        type=str,
        default=None,
        help="Path to prompt policy model (for hierarchical)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of episodes per strategy"
    )
    args = parser.parse_args()
    
    # Comparison mode
    if args.compare:
        df, detailed_df = run_comparison_benchmark(
            num_episodes=args.episodes,
            single_step_model=args.single_step_model,
            multi_step_model=args.multi_step_model
        )
        if detailed_df is not None:
            analyze_results(detailed_df)
    else:
        # Original benchmark logic
        cfg = load_config(args.config)
        print(f"Loaded config: {args.config}")
        
        if cfg.ENV_MODE == "single_step":
            model_path = args.model or get_latest_model("single_step")
            if model_path:
                print(f"Using model: {model_path}")
            df, detailed_df = run_benchmark_single_step(cfg, model_path, args.episodes)
            
        elif cfg.ENV_MODE == "multi_step":
            model_path = args.model or get_latest_model("multi_step")
            if model_path:
                print(f"Using model: {model_path}")
            df, detailed_df = run_benchmark_multistep(cfg, model_path, args.episodes)
            
        else:  # hierarchical
            struct_path = args.structure_model or get_latest_model(policy_type="structure")
            prompt_path = args.prompt_model or get_latest_model(policy_type="prompt")
            
            if struct_path:
                print(f"Using structure model: {struct_path}")
            if prompt_path:
                print(f"Using prompt model: {prompt_path}")
                
            df, detailed_df = run_benchmark_hierarchical(
                cfg, struct_path, prompt_path, args.episodes
            )
        
        if detailed_df is not None:
            analyze_results(detailed_df)
