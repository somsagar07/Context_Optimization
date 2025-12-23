from datasets import load_dataset
import abc
import random
import re

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

class GSM8kDataset(BaseDataset):
    """Grade School Math 8K dataset."""
    
    def __init__(self, split="train"):
        print(f"Loading GSM8k ({split})...")
        self.data = load_dataset("gsm8k", "main", split=split)
    
    def get_sample(self):
        idx = random.randint(0, len(self.data) - 1)
        sample = self.data[idx]
        return sample['question'], sample['answer']

    def extract_number(self, text: str):
        """Extract numerical answer from text (GSM8k specific)."""
        if "####" in text:
            text = text.split("####")[-1]
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(',', ''))
        return float(nums[-1]) if nums else None

    def evaluate_correctness(self, prediction: str, ground_truth: str) -> float:
        pred_num = self.extract_number(prediction)
        truth_num = self.extract_number(ground_truth)
        
        if pred_num is None or truth_num is None:
            return 0.0
        return 1.0 if abs(pred_num - truth_num) < 1e-3 else 0.0


class HotPotQADataset(BaseDataset):
    """HotPotQA dataset for multi-hop reasoning."""
    
    def __init__(self, split="train"):
        print(f"Loading HotPotQA ({split})...")
        self.data = load_dataset("hotpot_qa", "fullwiki", split=split)
    
    def get_sample(self):
        idx = random.randint(0, len(self.data) - 1)
        sample = self.data[idx]
        return sample['question'], sample['answer']

    def evaluate_correctness(self, prediction: str, ground_truth: str) -> float:
        # Simple exact match (case insensitive)
        pred = prediction.lower().strip()
        truth = ground_truth.lower().strip()
        
        if truth in pred or pred in truth:
            return 1.0
        return 0.0


def get_dataset_loader(name: str):
    """Factory function to get the right dataset loader."""
    if name == "gsm8k":
        return GSM8kDataset()
    elif name == "hotpotqa":
        return HotPotQADataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")

