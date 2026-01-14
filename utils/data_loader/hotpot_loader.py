from datasets import load_dataset
import random
from .base_loader import BaseDataset


class HotPotQADataset(BaseDataset):
    """HotPotQA dataset for multi-hop reasoning."""
    
    def __init__(self, split="train"):
        print(f"Loading HotPotQA ({split})...")
        self.data = load_dataset("hotpot_qa", "fullwiki", split=split)
        self.name = "hotpot_qa"
    
    def get_sample(self):
        idx = random.randint(0, len(self.data) - 1)
        sample = self.data[idx]
        return sample['question'], sample['answer']

    def evaluate_correctness(self, prediction: str, ground_truth: str) -> float:
        # Simple exact match (case insensitive)
        pred = prediction.lower().strip()
        truth = ground_truth.lower().strip()
        
        # Empty prediction is always incorrect
        if not pred:
            return 0.0
        
        if truth in pred or pred in truth:
            return 1.0
        return 0.0

