"""
Evaluation script for RL learned controller.
Supports both single-step and multi-step environments, with optional prompt learning.

Usage:
    # Use latest model with default HuggingFace model
    python scripts/eval_rl.py --config multi_step
    
    # Specify model path
    python scripts/eval_rl.py --config single_step --model path/to/model
    
    # Evaluate with prompt learning (must match how model was trained!)
    python scripts/eval_rl.py --config multi_step --learn-prompts --model path/to/model_prompts
    
    # More evaluation episodes
    python scripts/eval_rl.py --config multi_step --episodes 100
    
    # Use specific HuggingFace model
    python scripts/eval_rl.py --config single_step --hf-model Qwen/Qwen2.5-7B-Instruct
    
    # Use OpenRouter API
    python scripts/eval_rl.py --config single_step --api --api-model openai/gpt-4o
    
    # Evaluate on different dataset
    python scripts/eval_rl.py --config multi_step --dataset gsm8k --episodes 50
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from configs import load_config
from env import GeneralAgentEnv, MultiStepAgentEnv


def make_env(cfg, is_eval, use_api, api_model, hf_model, env_mode, learn_prompts=False):
    """Factory function to create environment (avoids lambda closure issues)."""
    def _init():
        if env_mode == "multi_step":
            return MultiStepAgentEnv(
                cfg=cfg,
                is_eval=is_eval,
                use_api=use_api,
                api_model=api_model,
                hf_model=hf_model,
                learn_prompts=learn_prompts
            )
        else:
            return GeneralAgentEnv(
                cfg=cfg,
                is_eval=is_eval,
                use_api=use_api,
                api_model=api_model,
                hf_model=hf_model
            )
    return _init


def get_latest_model():
    """Find the latest model in the models/flat_rl directory (recursively searches subfolders)."""
    # Look in project root models/flat_rl/ directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models", "flat_rl")
    
    # Search recursively in subfolders (models/flat_rl/{model_name}/*.zip)
    list_of_files = glob.glob(os.path.join(models_dir, "**", "*.zip"), recursive=True)
    
    if not list_of_files:
        # Also check flat directory for backwards compatibility
        list_of_files = glob.glob(os.path.join(models_dir, "*.zip"))
        
    if not list_of_files:
        # Also check models/ for backwards compatibility
        models_dir_fallback = os.path.join(project_root, "models")
        list_of_files = glob.glob(os.path.join(models_dir_fallback, "*.zip"))
        if not list_of_files:
            return None
            
    latest_file = max(list_of_files, key=os.path.getctime)
    # Return path without extension as PPO.load expects
    return os.path.splitext(latest_file)[0]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate RL Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="multi_step",
        choices=["single_step", "multi_step"],
        help="Configuration to use (must match how model was trained)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (without .zip). If not provided, uses latest model."
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="30",
        help="Number of episodes to evaluate, or 'all' for all datapoints"
    )
    from utils import validate_dataset_name, get_dataset_help_text
    
    parser.add_argument(
        "--dataset",
        type=validate_dataset_name,
        default=None,
        help=get_dataset_help_text()
    )
    
    # LLM configuration
    parser.add_argument(
        "--api",
        action="store_true",
        default=False,
        help="Use OpenRouter API instead of local HuggingFace model"
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default=None,
        help="OpenRouter model ID (e.g., 'openai/gpt-4o'). Required if --api is used."
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="HuggingFace model name. Uses config default if not specified."
    )
    
    # Prompt learning (multi_step only)
    parser.add_argument(
        "--learn-prompts",
        action="store_true",
        default=False,
        help="Enable prompt learning mode (multi_step only). Must match how model was trained!"
    )
    
    # Parallel workers (for API evaluation)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for evaluation (only works with --api mode)"
    )
    return parser.parse_args()


def run_single_episode(ep, rl_model, env, sample_idx=None):
    """
    Run a single evaluation episode.
    
    Args:
        ep: Episode number
        rl_model: Trained PPO model
        env: Environment instance (unwrapped, not VecEnv)
        sample_idx: If provided, use this specific dataset index
        
    Returns:
        Dict with episode results
    """
    # Reset environment and optionally use specific sample
    if sample_idx is not None and hasattr(env, 'dataset'):
        # Manually set the sample for deterministic evaluation
        dataset = env.dataset
        sample = dataset.data[sample_idx]
        dataset_name = getattr(dataset, 'name', '').lower()
        
        # Extract question and answer based on dataset type
        if dataset_name == 'gaia':
            question = sample['Question']
            answer = sample['Final answer']
            rel_path = sample.get('file_path', '')
            if rel_path and hasattr(dataset, 'data_dir'):
                full_path = os.path.join(dataset.data_dir, rel_path)
                question += f"\n\n[System Notification]\nFile Attachment: {full_path}\nYou can use your tools to read or process this file."
        elif dataset_name == 'medqa':
            data = sample['data']
            question = data['Question']
            options = data['Options']
            options_text = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
            question = f"{question}\n\nOptions:\n{options_text}"
            answer = data['Correct Answer']
        elif dataset_name == 'aime25':
            question = sample.get('problem', '')
            answer = sample.get('answer', '')
        elif dataset_name == 'drop':
            # DROP has passage + question structure (same format as get_sample())
            passage = sample.get('passage', '')
            question_text = sample.get('question', '')
            # Format exactly like DROPDataset.get_sample() does
            question = f"{passage}\n\nQuestion: {question_text}"
            # Extract answer from answers_spans (same logic as get_sample())
            answer_spans = sample.get('answers_spans', {})
            if 'spans' in answer_spans and len(answer_spans['spans']) > 0:
                # Use first answer (usually there's one primary answer)
                answer = answer_spans['spans'][0]
            else:
                # Fallback: try to find answer in other fields
                answer = sample.get('answer', '')
        else:
            # GSM8K, HotPotQA, and other standard datasets
            question = sample.get('question', sample.get('problem', ''))
            answer = sample.get('answer', '')
        
        env.current_q = question
        env.current_a = answer
        
        # Get observation based on environment type
        if hasattr(env, '_get_observation'):
            # MultiStepAgentEnv: need to set embedding and reset stage, then get observation
            env.question_embedding = env.worker.get_embedding(question)
            env.stage = 0
            env.workflow_depth = None
            env.agent1_tools = None
            env.agent1_budget = None
            env.agent2_tools = None
            env.agent2_budget = None
            env.answerer_budget = None
            if hasattr(env, 'reasoner_prompt'):
                env.reasoner_prompt = None
                env.verifier_prompt = None
                env.answerer_prompt = None
            obs = env._get_observation()
        else:
            # GeneralAgentEnv: observation is just the embedding
            obs = env.worker.get_embedding(question)
    else:
        obs, _ = env.reset()
    
    done = False
    episode_reward = 0.0
    decision_steps = 0
    
    while not done:
        # Add batch dimension for model prediction
        obs_batch = np.expand_dims(obs, axis=0)
        action, _ = rl_model.predict(obs_batch, deterministic=True)
        action = action[0] if isinstance(action, np.ndarray) and len(action.shape) > 0 else action
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        decision_steps += 1
        done = terminated or truncated
    
    return {
        "episode": ep,
        "correct": info.get("correct", False),
        "workflow": info.get("workflow", "Unknown"),
        "llm_steps": info.get("steps_taken", 0),
        "decision_steps": decision_steps,
        "tools": info.get("tools_used", 0),
        "tokens": info.get("total_tokens", 0),
        "reward": episode_reward,
        "agent1_tools": str(info.get("agent1_tools", [])),
        "agent2_tools": str(info.get("agent2_tools", [])),
        "agent1_budget": info.get("agent1_budget", "N/A"),
        "agent2_budget": info.get("agent2_budget", "N/A"),
        "answerer_budget": info.get("answerer_budget", "N/A"),
        "reasoner_prompt": info.get("reasoner_prompt"),
        "verifier_prompt": info.get("verifier_prompt"),
        "answerer_prompt": info.get("answerer_prompt"),
        # Add question, prediction, and ground truth for debugging
        "question": info.get("query", ""),
        "prediction": info.get("final_answer", ""),
        "ground_truth": info.get("ground_truth", ""),
    }


def run_eval_parallel(cfg, model_path: str, num_episodes: int, dataset_override: str = None,
                      use_api: bool = False, api_model: str = None, hf_model: str = None,
                      learn_prompts: bool = False, num_workers: int = 4):
    """
    Run parallel evaluation for RL controller (for API mode).
    
    Args:
        cfg: Configuration module
        model_path: Path to trained model
        num_episodes: Number of episodes
        dataset_override: Optional dataset override
        use_api: If True, use OpenRouter API
        api_model: OpenRouter model ID
        hf_model: HuggingFace model name
        learn_prompts: If True, enable prompt learning mode
        num_workers: Number of parallel workers
    """
    from utils.get_dataset import get_dataset_loader
    
    dataset_name = dataset_override or cfg.DATASET_NAME
    
    print(f"\n{'='*70}")
    print(f"EVALUATION: RL Learned Controller (Parallel)")
    print(f"{'='*70}")
    print(f"  Config:     {cfg.ENV_MODE}")
    print(f"  Dataset:    {dataset_name}")
    print(f"  Episodes:   {num_episodes}")
    print(f"  Workers:    {num_workers}")
    print(f"  Model:      {os.path.basename(model_path)}")
    if learn_prompts:
        print(f"  Prompts:    Learning enabled")
    print(f"  LLM:        API - {api_model}")
    print(f"{'='*70}\n")
    
    # Load dataset ONCE and share across all environments
    print("Loading dataset...")
    shared_dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=True)
    dataset_size = len(shared_dataset.tasks) if hasattr(shared_dataset, 'tasks') else len(shared_dataset.data)
    print(f"  Dataset loaded: {dataset_size} samples")
    
    if num_episodes > dataset_size:
        print(f"  Warning: Requested {num_episodes} episodes but only {dataset_size} samples.")
        num_episodes = dataset_size
    
    # Load the model once
    real_path = model_path if model_path.endswith('.zip') else model_path + ".zip"
    if not os.path.exists(real_path):
        print(f"Error: Model file not found: {real_path}")
        return None
    
    print(f"Loading model: {real_path}")
    rl_model = PPO.load(model_path, device='cpu')
    model_obs_shape = rl_model.observation_space.shape
    print(f"  Model observation space: {model_obs_shape}")
    
    # Pre-create environments for each worker
    print(f"\nPre-creating {num_workers} environments...")
    env_pools = []
    for i in range(num_workers):
        print(f"  Creating environment {i+1}/{num_workers}...")
        if cfg.ENV_MODE == "multi_step":
            env = MultiStepAgentEnv(
                cfg=cfg,
                is_eval=True,
                use_api=use_api,
                api_model=api_model,
                hf_model=hf_model,
                learn_prompts=learn_prompts
            )
        else:
            env = GeneralAgentEnv(
                cfg=cfg,
                is_eval=True,
                use_api=use_api,
                api_model=api_model,
                hf_model=hf_model
            )
        env.dataset = shared_dataset  # Share dataset
        env_pools.append(env)
    
    # Check observation space matches
    env_obs_shape = env_pools[0].observation_space.shape
    print(f"  Environment observation space: {env_obs_shape}")
    
    if model_obs_shape != env_obs_shape:
        print(f"\n  ⚠️  OBSERVATION SPACE MISMATCH!")
        print(f"    Model expects {model_obs_shape}, environment provides {env_obs_shape}")
        if model_obs_shape[0] == 1154:
            print(f"    → Model was trained WITH --learn-prompts, you MUST use --learn-prompts")
        elif model_obs_shape[0] == 1133:
            print(f"    → Model was trained WITHOUT --learn-prompts")
        return None
    
    # Pre-initialize embedders
    print(f"\n  Pre-initializing embedders...")
    for i, env in enumerate(env_pools):
        if hasattr(env, 'worker') and hasattr(env.worker, 'embedder'):
            embedder = env.worker.embedder
            if hasattr(embedder, '_init_embedder') and not embedder._initialized:
                embedder._init_embedder()
    print(f"  ✓ Embedders initialized")
    
    # Thread locks for each environment
    env_locks = [threading.Lock() for _ in range(num_workers)]
    
    # Results containers (thread-safe)
    results_lock = threading.Lock()
    all_results = []
    completed = [0]
    correct_count = [0]
    total_reward = [0.0]
    
    def worker_fn(ep):
        """Worker function to run a single episode."""
        env_idx = ep % num_workers
        env = env_pools[env_idx]
        
        with env_locks[env_idx]:
            return run_single_episode(ep, rl_model, env, sample_idx=ep)
    
    print(f"\nEvaluating {num_episodes} episodes with {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_fn, ep): ep for ep in range(num_episodes)}
        
        with tqdm(total=num_episodes, desc=f"Evaluating ({num_workers} workers)") as pbar:
            for future in as_completed(futures):
                ep = futures[future]
                try:
                    result = future.result()
                    
                    with results_lock:
                        all_results.append(result)
                        completed[0] += 1
                        correct_count[0] += int(result["correct"])
                        total_reward[0] += result["reward"]
                        
                        acc = correct_count[0] / completed[0] * 100
                        avg_rew = total_reward[0] / completed[0]
                        pbar.set_postfix({"acc": f"{acc:.1f}%", "reward": f"{avg_rew:.2f}"})
                        pbar.update(1)
                        
                except Exception as e:
                    print(f"\nError in episode {ep}: {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)
    
    # Sort by episode number
    all_results.sort(key=lambda x: x["episode"])
    
    # Create DataFrame
    detailed_df = pd.DataFrame(all_results)
    
    # Print results
    avg_acc = detailed_df["correct"].mean()
    avg_reward = detailed_df["reward"].mean()
    avg_steps = detailed_df["llm_steps"].mean()
    avg_tools = detailed_df["tools"].mean()
    avg_tokens = detailed_df["tokens"].mean()
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  Accuracy:           {avg_acc:.1%} ({detailed_df['correct'].sum()}/{num_episodes})")
    print(f"  Avg Reward:         {avg_reward:.3f}")
    print(f"  Avg LLM Steps:      {avg_steps:.2f}")
    print(f"  Avg Tools Used:     {avg_tools:.2f}")
    print(f"  Avg Tokens:         {avg_tokens:.0f}")
    print("="*70)
    
    # Workflow distribution
    print("\nWorkflow Distribution:")
    print(detailed_df["workflow"].value_counts().to_string())
    
    # Save results
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_folder = "default"
    if model_path:
        path_parts = os.path.normpath(model_path).split(os.sep)
        if "flat_rl" in path_parts:
            idx = path_parts.index("flat_rl")
            if idx + 1 < len(path_parts) - 1:
                model_folder = path_parts[idx + 1]
    
    log_dir = os.path.join(project_root, "logs", "eval", model_folder)
    os.makedirs(log_dir, exist_ok=True)
    
    output_filename = f"eval_results_{cfg.ENV_MODE}_{dataset_name}_{int(time.time())}.csv"
    output_path = os.path.join(log_dir, output_filename)
    detailed_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return detailed_df


def run_eval(cfg, model_path: str = None, num_episodes: int = 30, dataset_override: str = None,
              use_api: bool = False, api_model: str = None, hf_model: str = None,
              learn_prompts: bool = False):
    """
    Run evaluation for RL controller.
    
    Args:
        cfg: Configuration module
        model_path: Path to trained model (without .zip extension)
        num_episodes: Number of episodes
        dataset_override: Optional dataset override
        use_api: If True, use OpenRouter API
        api_model: OpenRouter model ID
        hf_model: HuggingFace model name
        learn_prompts: If True, enable prompt learning mode (multi_step only)
    """
    dataset_name = dataset_override or cfg.DATASET_NAME
    
    # Resolve model path
    if model_path is None:
        model_path = get_latest_model()
        if model_path is None:
            print("Error: No model found in models/ directory.")
            return None
            
    print(f"\n{'='*70}")
    print(f"EVALUATION: RL Learned Controller")
    print(f"{'='*70}")
    print(f"  Config:     {cfg.ENV_MODE}")
    print(f"  Dataset:    {dataset_name}")
    print(f"  Episodes:   {num_episodes}")
    print(f"  Model:      {os.path.basename(model_path)}")
    if learn_prompts:
        print(f"  Prompts:    Learning enabled")
    if use_api:
        print(f"  LLM:        API - {api_model}")
    else:
        print(f"  LLM:        HuggingFace - {hf_model or 'default'}")
    print(f"{'='*70}\n")
    
    # Setup environment based on mode
    print(f"Using {'MultiStepAgentEnv' if cfg.ENV_MODE == 'multi_step' else 'GeneralAgentEnv'}")
    print(f"  learn_prompts={learn_prompts}")
    env = DummyVecEnv([make_env(cfg, is_eval=True, use_api=use_api, 
                                 api_model=api_model, hf_model=hf_model, 
                                 env_mode=cfg.ENV_MODE, learn_prompts=learn_prompts)])
    print(f"  Environment observation space: {env.observation_space.shape}")
    
    # Load RL model with the environment (needed for correct observation space)
    real_path = model_path if model_path.endswith('.zip') else model_path + ".zip"
    if not os.path.exists(real_path):
        print(f"Error: Model file not found: {real_path}")
        return None
        
    print(f"Loading model: {real_path}")
    
    # First load model WITHOUT env to check its original observation space
    rl_model_check = PPO.load(model_path, device='cpu')
    model_obs_shape = rl_model_check.observation_space.shape
    env_obs_shape = env.observation_space.shape
    print(f"  Model trained with obs space: {model_obs_shape}")
    print(f"  Eval env observation space:   {env_obs_shape}")
    
    if model_obs_shape != env_obs_shape:
        print(f"\n  ⚠️  OBSERVATION SPACE MISMATCH!")
        print(f"    Model expects {model_obs_shape}, environment provides {env_obs_shape}")
        # Detect if it's a learn_prompts mismatch
        # With learn_prompts=True:  1024 + 7 + 9 + 48 + 48 + 7 + 6 + 5 = 1154
        # With learn_prompts=False: 1024 + 4 + 9 + 48 + 48 = 1133
        if model_obs_shape[0] == 1154 and env_obs_shape[0] != 1154:
            print(f"    → Model was trained WITH --learn-prompts, you MUST use --learn-prompts during eval")
            return None
        elif model_obs_shape[0] == 1133 and env_obs_shape[0] != 1133:
            print(f"    → Model was trained WITHOUT --learn-prompts, do NOT use --learn-prompts during eval")
            return None
        else:
            print(f"    → Check that --config and other args match training configuration")
            return None
    
    del rl_model_check  # Free memory
    
    # Now load with env for proper prediction
    rl_model = PPO.load(model_path, env=env, device='cpu')
    
    # Results container
    detailed_results = []
    
    print("\nRunning evaluation...", flush=True)
    start_time = time.time()
    
    accuracies = []
    steps_list = []
    tools_list = []
    tokens_list = []
    workflows = []
    rewards_list = []
    decision_steps_list = []  # Track multi-step decision count
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        decision_steps = 0
        
        # Multi-step loop: keep stepping until episode terminates
        while not done:
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            episode_reward += float(rewards[0])
            decision_steps += 1
            done = dones[0]
        
        # Extract final info (only available when episode ends)
        info = infos[0]
        
        accuracies.append(1 if info["correct"] else 0)
        steps_list.append(info["steps_taken"])  # LLM calls
        tools_list.append(info["tools_used"])
        tokens_list.append(info["total_tokens"])
        workflows.append(info["workflow"])
        rewards_list.append(episode_reward)
        decision_steps_list.append(decision_steps)
        
        # Store detailed result
        result_entry = {
            "episode": ep,
            "correct": info["correct"],
            "workflow": info["workflow"],
            "llm_steps": info["steps_taken"],
            "decision_steps": decision_steps,  # RL decisions made
            "tools": info["tools_used"],
            "tokens": info["total_tokens"],
            "reward": episode_reward,
            "agent1_tools": str(info.get("agent1_tools", [])),
            "agent2_tools": str(info.get("agent2_tools", [])),
            "agent1_budget": info.get("agent1_budget", "N/A"),
            "agent2_budget": info.get("agent2_budget", "N/A"),
            "answerer_budget": info.get("answerer_budget", "N/A"),
            # Add question, prediction, and ground truth for debugging
            "question": info.get("query", ""),
            "prediction": info.get("final_answer", ""),
            "ground_truth": info.get("ground_truth", ""),
        }
        
        # Add prompt info if learning prompts
        if "reasoner_prompt" in info:
            result_entry["reasoner_prompt"] = info["reasoner_prompt"]
            result_entry["verifier_prompt"] = info.get("verifier_prompt")
            result_entry["answerer_prompt"] = info.get("answerer_prompt")
        
        detailed_results.append(result_entry)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes} - Running Acc: {np.mean(accuracies):.1%}")
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    avg_acc = np.mean(accuracies)
    avg_steps = np.mean(steps_list)
    avg_decision_steps = np.mean(decision_steps_list)
    avg_tools = np.mean(tools_list)
    avg_tokens = np.mean(tokens_list)
    avg_reward = np.mean(rewards_list)
    
    # Print results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  Accuracy:           {avg_acc:.1%} ({sum(accuracies)}/{num_episodes})")
    print(f"  Avg Reward:         {avg_reward:.3f}")
    print(f"  Avg LLM Steps:      {avg_steps:.2f}")
    print(f"  Avg Decision Steps: {avg_decision_steps:.2f}")
    print(f"  Avg Tools Used:     {avg_tools:.2f}")
    print(f"  Avg Tokens:         {avg_tokens:.0f}")
    print(f"  Time:               {elapsed:.1f}s")
    print("="*70)
    
    # Workflow distribution
    detailed_df = pd.DataFrame(detailed_results)
    print("\nWorkflow Distribution:")
    print(detailed_df["workflow"].value_counts().to_string())
    
    # Save detailed results to logs/eval/{model_folder}/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Determine model folder from model_path (e.g., models/flat_rl/gpt-4o/model.zip -> gpt-4o)
    model_folder = "default"
    if model_path:
        # Extract folder name from path
        path_parts = os.path.normpath(model_path).split(os.sep)
        if "flat_rl" in path_parts:
            idx = path_parts.index("flat_rl")
            if idx + 1 < len(path_parts) - 1:  # There's a subfolder between flat_rl and the file
                model_folder = path_parts[idx + 1]
    
    log_dir = os.path.join(project_root, "logs", "eval", model_folder)
    os.makedirs(log_dir, exist_ok=True)
    
    output_filename = f"eval_results_{cfg.ENV_MODE}_{dataset_name}_{int(time.time())}.csv"
    output_path = os.path.join(log_dir, output_filename)
    detailed_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    return detailed_df


def analyze_results(detailed_df: pd.DataFrame):
    """Analyze evaluation results."""
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Statistics
    print("\nStatistics:")
    print(f"  Correct:  {detailed_df['correct'].sum()} / {len(detailed_df)}")
    print(f"  Accuracy: {detailed_df['correct'].mean():.1%}")
    print(f"  Reward:   {detailed_df['reward'].mean():.3f} ± {detailed_df['reward'].std():.3f}")
    print(f"  Tokens:   {detailed_df['tokens'].mean():.0f} ± {detailed_df['tokens'].std():.0f}")
    
    # Workflow distribution
    print("\nWorkflow Distribution:")
    workflow_dist = detailed_df["workflow"].value_counts()
    for workflow, count in workflow_dist.items():
        pct = count / len(detailed_df) * 100
        print(f"  {workflow}: {count} ({pct:.1f}%)")
    
    # Accuracy by workflow
    print("\nAccuracy by Workflow:")
    for workflow in detailed_df["workflow"].unique():
        subset = detailed_df[detailed_df["workflow"] == workflow]
        acc = subset["correct"].mean()
        print(f"  {workflow}: {acc:.1%} (n={len(subset)})")


if __name__ == "__main__":
    args = parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    print(f"Loaded config: {args.config}")
    
    # Override dataset if specified
    if args.dataset:
        cfg.DATASET_NAME = args.dataset
    
    # Handle "all" episodes - evaluate on entire dataset
    if args.episodes.lower() == "all":
        from utils.get_dataset import get_dataset_loader
        dataset = get_dataset_loader(cfg.DATASET_NAME, is_eval=True)
        # Get dataset size - different datasets use different attributes
        if hasattr(dataset, 'tasks'):
            num_episodes = len(dataset.tasks)
        elif hasattr(dataset, 'data'):
            num_episodes = len(dataset.data)
        else:
            num_episodes = len(dataset)
        print(f"Evaluating on ALL {num_episodes} datapoints from {cfg.DATASET_NAME}")
    else:
        num_episodes = int(args.episodes)
    
    # Validate LLM configuration
    if args.api and not args.api_model:
        print("Error: --api-model is required when using --api")
        sys.exit(1)
    
    # Validate learn_prompts (only works with multi_step)
    if args.learn_prompts and args.config != "multi_step":
        print("Error: --learn-prompts is only supported with --config multi_step")
        sys.exit(1)
    
    # Run evaluation (parallel for API mode with workers > 1)
    if args.api and args.workers > 1:
        print(f"Using parallel evaluation with {args.workers} workers")
        detailed_df = run_eval_parallel(
            cfg=cfg,
            model_path=args.model,
            num_episodes=num_episodes,
            dataset_override=args.dataset,
            use_api=args.api,
            api_model=args.api_model,
            hf_model=args.hf_model,
            learn_prompts=args.learn_prompts,
            num_workers=args.workers
        )
    else:
        if args.workers > 1 and not args.api:
            print("Warning: Parallel workers only supported with --api mode. Using sequential evaluation.")
        detailed_df = run_eval(
            cfg=cfg,
            model_path=args.model,
            num_episodes=num_episodes,
            dataset_override=args.dataset,
            use_api=args.api,
            api_model=args.api_model,
            hf_model=args.hf_model,
            learn_prompts=args.learn_prompts
        )
    
    if detailed_df is not None:
        analyze_results(detailed_df)
