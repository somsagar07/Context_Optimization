import json
import os
import random
import requests
from .base_loader import BaseDataset

class Tau2Dataset(BaseDataset):
    """
    Loader for Tau-2 Bench (Retail, Airline, Telecom).
    Returns task_id along with question for tau2 execution.
    """
    GITHUB_BASE = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains"
    
    def __init__(self, split="test", domain="retail", cache_dir="./data_cache/tau2"):
        self.name = f"tau2_{domain}"
        self.domain = domain
        self.split = split
        self.cache_dir = cache_dir
        
        # Don't create cache_dir here - we'll use the local repo if available
        # os.makedirs(self.cache_dir, exist_ok=True)
        
        raw_tasks = self._load_or_fetch_data()
        self.tasks = [self._normalize_task(t) for t in raw_tasks]
        print(f"Loaded and normalized {len(self.tasks)} tasks for Tau-2 {domain} (split: {split})")

    def __len__(self):
        """Return the number of tasks in the dataset."""
        return len(self.tasks)
    
    def __iter__(self):
        """
        Make the dataset iterable for prompt generation.
        Returns items compatible with prompt generator expectations.
        """
        for task in self.tasks:
            # Format for prompt generator (simplified, no internal instructions)
            question = f"Customer interaction: {task['user_goal']}"
            
            # Extract action names from actions list (actions might be dicts or strings)
            actions = task.get('actions', [])
            if actions:
                # If actions are dicts, extract the 'name' field
                if isinstance(actions[0], dict):
                    action_names = [a.get('name', str(a)) for a in actions]
                else:
                    action_names = [str(a) for a in actions]
                actions_str = ", ".join(action_names)
            else:
                actions_str = "Complete the customer interaction"
            
            answer = f"Expected outcome: {actions_str}"
            
            yield {
                'question': question,
                'answer': answer,
                'input': question,
                'output': answer,
                'task_id': task.get('task_id', 'Unknown')
            }
    
    def _find_tau2_data_root(self):
        """Find the tau2_data_root directory (local repository)."""
        # Check TAU2_DATA_DIR environment variable first
        if 'TAU2_DATA_DIR' in os.environ:
            tau2_data_dir = os.environ['TAU2_DATA_DIR']
            if os.path.exists(tau2_data_dir):
                return tau2_data_dir
        
        # Check common locations relative to this file
        possible_paths = [
            './data_cache/tau2_data_root',
            '../data_cache/tau2_data_root',
            os.path.join(os.path.dirname(__file__), '../../data_cache/tau2_data_root'),
            os.path.join(os.path.dirname(__file__), '../../../data_cache/tau2_data_root'),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        return None

    def _load_or_fetch_data(self):
        """
        Load data with priority:
        1. Local tau2 repository (tau2_data_root/tau2/domains/{domain}/tasks.json) - SINGLE SOURCE OF TRUTH
        2. Cache directory (cache_dir/{domain}_tasks.json)
        3. Download from GitHub as last resort
        """
        # Check local tau2 repository (single source of truth)
        tau2_data_root = self._find_tau2_data_root()
        if tau2_data_root:
            repo_tasks_path = os.path.join(tau2_data_root, 'tau2', 'domains', self.domain, 'tasks.json')
            if os.path.exists(repo_tasks_path):
                print(f"Found tasks in local tau2 repository: {repo_tasks_path}")
                try:
                    with open(repo_tasks_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error reading {repo_tasks_path}: {e}")
                    # Continue to next priority
        
        # Check cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        candidates = [
            f"{self.domain}_tasks.json",
            f"{self.domain}.json",
            "tasks.json"
        ]
        
        for fname in candidates:
            path = os.path.join(self.cache_dir, fname)
            if os.path.exists(path):
                print(f"Found cached file: {path}")
                try:
                    with open(path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    # Continue to next candidate

        # Download from GitHub as last resort
        url = f"{self.GITHUB_BASE}/{self.domain}/tasks.json"
        save_path = os.path.join(self.cache_dir, f"{self.domain}_tasks.json")
        
        print(f"Downloading from GitHub: {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            with open(save_path, 'w') as f:
                json.dump(data, f)
            print(f"Saved to {save_path}")
            return data
        except Exception as e:
            raise RuntimeError(
                f"Failed to download from GitHub ({url}).\n"
                f"Error: {e}\n"
                "Please check your internet connection or if the repo structure has changed."
            )

    def _normalize_task(self, raw_task):
        scenario = raw_task.get("user_scenario", {})
        instructions = scenario.get("instructions", {})
        
        goal = instructions.get("reason_for_call")
        if not goal:
            goal = (
                raw_task.get("description") or 
                raw_task.get("goal") or 
                "No explicit goal found."
            )

        known = instructions.get("known_info", "")
        unknown = instructions.get("unknown_info", "")
        task_notes = instructions.get("task_instructions", "")

        profile = {
            "Identity": known,
            "Constraints": unknown, 
            "Persona Notes": task_notes
        }

        eval_criteria = raw_task.get("evaluation_criteria", {})
        actions = eval_criteria.get("actions", [])

        return {
            "user_goal": goal,
            "task_id": raw_task.get("id", "Unknown"),
            "user_profile": profile,
            "actions": actions,
            "raw": raw_task
        }

    def get_sample(self):
        """
        Returns (question, task_object) for tau2.
        
        For tau2, the question should be based on the user's stated goal (reason_for_call),
        NOT the internal scenario instructions. The actual conversation will be handled
        by the tau2 gym environment which provides the proper initial message.
        """
        idx = random.randint(0, len(self.tasks) - 1)
        task = self.tasks[idx]
        
        # For tau2, use only the user's stated goal (reason_for_call)
        # This is what the agent would see in a real scenario
        # The gym environment will provide the actual initial message during execution
        question = (
            f"Customer interaction in {self.domain.capitalize()} domain.\n"
            f"User's request: {task['user_goal']}"
        )
        
        return question, task

    def evaluate_correctness(self, prediction: str, ground_truth: dict) -> float:
        # For tau2, correctness is determined by Pass^k from execution wrapper
        # This is a placeholder - actual evaluation happens in execution wrapper
        return 0.0