from datasets import load_dataset
import random
import re
from .base_loader import BaseDataset

class GSM8kDataset(BaseDataset):
    """Grade School Math 8K dataset."""
    
    def __init__(self, split="train"):
        print(f"Loading GSM8k ({split})...")
        self.data = load_dataset("gsm8k", "main", split=split)
        self.name = "gsm8k"
    
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