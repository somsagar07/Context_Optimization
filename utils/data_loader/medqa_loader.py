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
        Robust evaluation that prioritizes the 'Final Answer' section 
        and Option Letters to avoid False Positives.
        """
        pred_text = prediction.strip()
        
        # 1. Parse Ground Truth
        # Assumes ground_truth format "A Answer Text" or just "Answer Text"
        gt_letter_match = re.match(r'^([A-D])\b', ground_truth)
        gt_letter = gt_letter_match.group(1) if gt_letter_match else None
        
        # Normalize the text part of ground truth for fallback comparison
        gt_text_normalized = re.sub(r'\W+', '', ground_truth.lower())
        if gt_letter:
            # Remove the leading "A " from the normalized text
            gt_text_normalized = re.sub(r'\W+', '', ground_truth[1:].lower())

        # 2. Extract "Final Answer" section from Prediction
        # Models often reason about wrong answers before giving the right one.
        # We only want to grade the final decision.
        final_answer_patterns = [
            r"Final Answer\s*[:\-\s](.*)",  # Matches "Final Answer: ..."
            r"The answer is\s*[:\-\s](.*)", # Matches "The answer is ..."
            r"Answer\s*[:\-\s](.*)"         # Matches "Answer: ..."
        ]
        
        extracted_answer = pred_text # Default to full text if no header found
        for pattern in final_answer_patterns:
            match = re.search(pattern, pred_text, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
                # If multiple matches, usually the last one is the real conclusion,
                # but regex search finds the first. Let's try `rsplit` logic if regex feels risky.
                # A safer split approach:
                if "Final Answer" in pred_text:
                    extracted_answer = pred_text.split("Final Answer")[-1]
                break

        # 3. Check for Option Letter in Extracted Answer (Primary Metric)
        if gt_letter:
            # Look for "A", "B", "C", "D" specifically in the final answer
            # We look for the letter followed by punctuation or end of string
            pred_letter_match = re.search(r'\b([A-D])\b', extracted_answer)
            if pred_letter_match:
                predicted_letter = pred_letter_match.group(1)
                return 1.0 if predicted_letter == gt_letter else 0.0

        # 4. Fallback: Strict Text Containment (if letter extraction failed)
        # We verify if the normalized ground truth text exists in the normalized extracted answer.
        pred_normalized = re.sub(r'\W+', '', extracted_answer.lower())
        
        if gt_text_normalized and gt_text_normalized in pred_normalized:
            return 1.0
            
        return 0.0
    