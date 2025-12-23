from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import json
import os
import time


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback for logging and saving training metrics.
    Saves full training history to JSON for later analysis.
    """
    
    def __init__(self, verbose=0, save_dir="logs", save_every_episodes=500):
        """
        Initialize the callback.
        
        Args:
            verbose: Verbosity level
            save_dir: Directory to save logs
            save_every_episodes: Save log every N episodes
        """
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.save_dir = save_dir
        self.save_every_episodes = save_every_episodes
        self.save_path = None
        
        # Full history (saved to disk)
        self.all_episodes = []
        
        # Running stats for display
        self.running_correct = 0
        self.running_total = 0
        
    def _on_training_start(self) -> None:
        """Initialize save path when training starts."""
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = int(time.time())
        self.save_path = os.path.join(self.save_dir, f"training_log_{timestamp}.json")
        print(f"Training log will be saved to: {self.save_path}")
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [])
        
        for idx, info in enumerate(infos):
            if "correct" in info:
                # Update running stats
                self.running_total += 1
                if info["correct"]:
                    self.running_correct += 1
                
                # Get reward for this episode (if available)
                reward = float(rewards[idx]) if idx < len(rewards) else 0.0
                
                # Store episode data
                episode_data = {
                    "step": self.num_timesteps,
                    "reward": reward,
                    "correct": info["correct"],
                    "workflow": info.get("workflow", ""),
                    "steps_taken": info.get("steps_taken", 0),
                    "tools_used": info.get("tools_used", 0),
                    "reasoner_tools": info.get("reasoner_tools", []),
                    "verifier_tools": info.get("verifier_tools", []),
                    "reasoner_budget": info.get("reasoner_budget", ""),
                    "verifier_budget": info.get("verifier_budget", ""),
                    "answerer_budget": info.get("answerer_budget", ""),
                    "total_tokens": info.get("total_tokens", 0),
                    "episode_length": info.get("episode_length", 1),  # For multi-step
                }
                self.all_episodes.append(episode_data)
                
                # Save periodically
                if len(self.all_episodes) % self.save_every_episodes == 0:
                    self._save_log()
            
        return True

    def _on_rollout_end(self) -> None:
        """Log metrics at end of each rollout."""
        if self.running_total > 0:
            accuracy = self.running_correct / self.running_total
            
            # Get recent stats (last 50 episodes)
            recent = self.all_episodes[-50:] if len(self.all_episodes) >= 50 else self.all_episodes
            if recent:
                avg_steps = np.mean([e["steps_taken"] for e in recent])
                avg_tools = np.mean([e["tools_used"] for e in recent])
                avg_tokens = np.mean([e["total_tokens"] for e in recent])
                avg_reward = np.mean([e["reward"] for e in recent])
                recent_acc = np.mean([1 if e["correct"] else 0 for e in recent])
                avg_ep_len = np.mean([e.get("episode_length", 1) for e in recent])
            else:
                avg_steps = avg_tools = avg_tokens = avg_reward = recent_acc = avg_ep_len = 0
            
            # Log to TensorBoard
            self.logger.record("custom/accuracy_total", accuracy)
            self.logger.record("custom/accuracy_recent", recent_acc)
            self.logger.record("custom/avg_reward", avg_reward)
            self.logger.record("custom/avg_steps", avg_steps)
            self.logger.record("custom/avg_tools", avg_tools)
            self.logger.record("custom/avg_tokens", avg_tokens)
            self.logger.record("custom/avg_episode_length", avg_ep_len)
            self.logger.record("custom/total_episodes", self.running_total)
            
            if self.verbose > 0:
                print(f"  Episodes: {self.running_total} | "
                      f"Acc (total): {accuracy:.1%} | "
                      f"Acc (recent): {recent_acc:.1%} | "
                      f"Reward: {avg_reward:.3f} | "
                      f"Steps: {avg_steps:.1f} | "
                      f"EpLen: {avg_ep_len:.1f} | "
                      f"Tokens: {avg_tokens:.0f}")

    def _on_training_end(self) -> None:
        """Save final log when training ends."""
        self._save_log()
        print(f"\nTraining complete! Log saved to: {self.save_path}")
        print(f"Total episodes: {len(self.all_episodes)}")
        if self.all_episodes:
            final_acc = np.mean([1 if e["correct"] else 0 for e in self.all_episodes])
            final_reward = np.mean([e["reward"] for e in self.all_episodes])
            print(f"Final accuracy: {final_acc:.1%}")
            print(f"Average reward: {final_reward:.3f}")
    
    def _save_log(self):
        """Save training log to JSON file."""
        if self.save_path and self.all_episodes:
            rewards = [e["reward"] for e in self.all_episodes]
            summary = {
                "total_episodes": len(self.all_episodes),
                "total_correct": sum(1 for e in self.all_episodes if e["correct"]),
                "accuracy": sum(1 for e in self.all_episodes if e["correct"]) / len(self.all_episodes),
                "avg_reward": float(np.mean(rewards)),
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "std_reward": float(np.std(rewards)),
                "episodes": self.all_episodes
            }
            with open(self.save_path, "w") as f:
                json.dump(summary, f, indent=2)
