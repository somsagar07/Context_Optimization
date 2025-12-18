from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback for logging training metrics.
    Tracks accuracy, tool usage, and token budgets.
    """
    
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.accuracy_history = []
        self.steps_history = []
        self.tools_history = []
        self.tokens_history = []
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        
        for info in infos:
            if "correct" in info:
                self.accuracy_history.append(1 if info["correct"] else 0)
            if "steps_taken" in info:
                self.steps_history.append(info["steps_taken"])
            if "tools_used" in info:
                self.tools_history.append(info["tools_used"])
            if "total_tokens" in info:
                self.tokens_history.append(info["total_tokens"])

        # Keep rolling window of last 100
        window = 100
        if len(self.accuracy_history) > window:
            self.accuracy_history = self.accuracy_history[-window:]
            self.steps_history = self.steps_history[-window:]
            self.tools_history = self.tools_history[-window:]
            self.tokens_history = self.tokens_history[-window:]
            
        return True

    def _on_rollout_end(self) -> None:
        if len(self.accuracy_history) > 0:
            avg_acc = np.mean(self.accuracy_history)
            avg_steps = np.mean(self.steps_history) if self.steps_history else 0
            avg_tools = np.mean(self.tools_history) if self.tools_history else 0
            avg_tokens = np.mean(self.tokens_history) if self.tokens_history else 0
            
            self.logger.record("custom/accuracy", avg_acc)
            self.logger.record("custom/avg_steps", avg_steps)
            self.logger.record("custom/avg_tools", avg_tools)
            self.logger.record("custom/avg_tokens", avg_tokens)
            
            if self.verbose > 0:
                print(f"  Accuracy: {avg_acc:.2%} | Steps: {avg_steps:.1f} | Tools: {avg_tools:.1f} | Tokens: {avg_tokens:.0f}")

