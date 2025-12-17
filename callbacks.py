from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard
    and displaying metrics in the progress bar.
    """
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.accuracy_history = []
        self.tool_usage_history = []
        self.context_history = []
        self.current_episode_rewards = []
        
    def _on_step(self) -> bool:
        # Access the info dict from the environment
        # self.locals['infos'] contains the info dicts for each environment
        infos = self.locals.get("infos", [{}])
        
        for info in infos:
            if "correct" in info:
                self.accuracy_history.append(1 if info["correct"] else 0)
            if "steps_taken" in info:
                self.tool_usage_history.append(info["steps_taken"]) # Reuse variable for steps
            if "context" in info:
                self.context_history.append(0) # Not used anymore

        # Keep history short for rolling average (last 100 steps)
        window = 100
        if len(self.accuracy_history) > window:
            self.accuracy_history = self.accuracy_history[-window:]
            self.tool_usage_history = self.tool_usage_history[-window:]
            self.context_history = self.context_history[-window:]
            
        return True

    def _on_rollout_end(self) -> None:
        # Log to TensorBoard or Print
        if len(self.accuracy_history) > 0:
            avg_acc = np.mean(self.accuracy_history)
            avg_steps = np.mean(self.tool_usage_history)
            
            self.logger.record("custom/accuracy", avg_acc)
            self.logger.record("custom/avg_steps", avg_steps)
            
            if self.verbose > 0:
                print(f"  Accuracy: {avg_acc:.2%} | Avg Steps: {avg_steps:.2f}")


