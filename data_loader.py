# data_loader.py
from datasets import load_dataset
import abc
import random
import re

class BaseDataset(abc.ABC):
    @abc.abstractmethod
    def get_sample(self):
        """Returns (question, answer)"""
        pass
    
    @abc.abstractmethod
    def evaluate_correctness(self, prediction: str, ground_truth: str) -> float:
        """Returns 1.0 for correct, 0.0 for incorrect"""
        pass

class GSM8kDataset(BaseDataset):
    def __init__(self, split="train"):
        print(f"Loading GSM8k ({split})...")
        self.data = load_dataset("gsm8k", "main", split=split)
    
    def get_sample(self):
        idx = random.randint(0, len(self.data) - 1)
        sample = self.data[idx]
        return sample['question'], sample['answer']

    def extract_number(self, text):
        # GSM8k specific extraction
        if "####" in text:
            text = text.split("####")[-1]
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(',', ''))
        return float(nums[-1]) if nums else None

    def evaluate_correctness(self, prediction, ground_truth):
        pred_num = self.extract_number(prediction)
        truth_num = self.extract_number(ground_truth)
        
        if pred_num is None or truth_num is None:
            return 0.0
        return 1.0 if abs(pred_num - truth_num) < 1e-3 else 0.0

# Factory function to get the right dataset
def get_dataset_loader(name: str):
    if name == "gsm8k":
        return GSM8kDataset()
    # elif name == "hotpotqa": return HotPotQADataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")