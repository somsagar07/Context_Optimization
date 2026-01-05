from datasets import load_dataset
import random
import re
from .base_loader import BaseDataset


class AIME25Dataset(BaseDataset):
    """AIME25 dataset for American Invitational Mathematics Examination problems."""
    
    def __init__(self, split="train"):
        print(f"Loading AIME25 ({split})...")
        # AIME25 only has "test" split with 30 examples
        # We'll split it ourselves: first 20 for train, last 10 for eval
        full_data = load_dataset("math-ai/aime25", split="test")
        self.name = "aime25"
        
        total_len = len(full_data)
        split_idx = 20  # Use first 20 for training, last 10 for evaluation
        
        if split == "train":
            print(f"AIME25: Using training split (0 to {split_idx})")
            self.data = full_data.select(range(0, split_idx))
        elif split == "test" or split == "validation":
            print(f"AIME25: Using evaluation split ({split_idx} to {total_len})")
            self.data = full_data.select(range(split_idx, total_len))
        else:
            # Use full dataset if unknown split
            self.data = full_data
    
    def get_sample(self):
        """Returns (question, answer) tuple."""
        idx = random.randint(0, len(self.data) - 1)
        sample = self.data[idx]
        
        # AIME25 structure: {'problem': ..., 'answer': ..., 'id': ...}
        question = sample['problem']
        answer = sample['answer']
        
        return question, answer
    
    def extract_number(self, text: str):
        """Extract numerical answer from text (AIME answers are integers)."""
        # AIME answers are usually integers, but may be in various formats
        # Look for numbers, especially at the end
        if "####" in text:
            text = text.split("####")[-1]
        
        # Try to find integers (AIME answers are 3-digit numbers 000-999)
        nums = re.findall(r'\b\d{1,4}\b', text.replace(',', ''))
        if nums:
            return int(nums[-1])
        
        # Fallback: any number
        nums = re.findall(r'[-+]?\d+', text.replace(',', ''))
        return int(nums[-1]) if nums else None
    
    def evaluate_correctness(self, prediction: str, ground_truth: str) -> float:
        """
        Evaluate correctness for AIME problems.
        AIME answers are  3-digit integers (000-999).
        """
        pred_num = self.extract_number(prediction)
        truth_num = self.extract_number(ground_truth)
        
        if pred_num is None or truth_num is None:
            return 0.0
        
        # Exact match for AIME (answers are specific integers)
        return 1.0 if pred_num == truth_num else 0.0

