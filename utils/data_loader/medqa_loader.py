from datasets import load_dataset
import random
import re
from .base_loader import BaseDataset


class MedQADataset(BaseDataset):
    """MedQA dataset for medical question answering (USMLE-style multiple choice)."""
    
    def __init__(self, split="train"):
        print(f"Loading MedQA ({split})...")
        # MedQA is available on HuggingFace as "openlifescienceai/medqa", It has splits: train, dev, test
        # We'll use "dev" as eval split and "train" for training
        self.data = load_dataset("openlifescienceai/medqa", split=split)
        self.name = "medqa"
    
    def get_sample(self):
        """Returns (question, answer) tuple."""
        idx = random.randint(0, len(self.data) - 1)
        sample = self.data[idx]
        
        # openlifescienceai/medqa structure: {'id': ..., 'data': {...}, 'subject_name': ...}
        # The actual data is in the 'data' field with keys: 'Question', 'Options', 'Correct Answer', 'Correct Option'
        data = sample['data']
        
        # Get question and options
        question = data['Question']
        options = data['Options']  # Dict: {'A': '...', 'B': '...', etc.}
        
        # Format question with options
        options_text = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
        question = f"{question}\n\nOptions:\n{options_text}"
        
        # Get correct answer (dataset always has 'Correct Answer')
        answer = data['Correct Answer']
        
        return question, answer
    
    def evaluate_correctness(self, prediction: str, ground_truth: str) -> float:
        """
        Evaluate correctness for MedQA (multiple choice).
        Handles letter answers (A, B, C, D) and full text matching.
        """
        pred = prediction.strip().upper()
        truth = ground_truth.strip().upper()
        
        # Extract letter answers (A, B, C, D, E, F)
        pred_letter = re.search(r'\b([A-F])\b', pred)
        truth_letter = re.search(r'\b([A-F])\b', truth)
        
        if pred_letter and truth_letter:
            return 1.0 if pred_letter.group(1) == truth_letter.group(1) else 0.0
        
        # Match full text (case-insensitive, normalized)
        pred_normalized = re.sub(r'[^\w\s]', '', pred.lower())
        truth_normalized = re.sub(r'[^\w\s]', '', truth.lower())
        
        if truth_normalized in pred_normalized or pred_normalized in truth_normalized:
            return 1.0
        
        return 0.0

