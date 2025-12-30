import abc

class BaseDataset(abc.ABC):
    """Abstract base class for datasets."""
    
    @abc.abstractmethod
    def get_sample(self):
        """Returns (question, answer) tuple."""
        pass
    
    @abc.abstractmethod
    def evaluate_correctness(self, prediction: str, ground_truth: str) -> float:
        """Returns 1.0 for correct, 0.0 for incorrect."""
        pass