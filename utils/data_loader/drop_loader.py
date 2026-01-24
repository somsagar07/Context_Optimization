from datasets import load_dataset
import random
import re
from typing import List, Union
from .base_loader import BaseDataset


class DROPDataset(BaseDataset):
    """
    DROP (Discrete Reasoning Over the content of Paragraphs) dataset.
    
    DROP is a reading comprehension dataset requiring numerical reasoning.
    Answers can be:
    - Numbers (integers or floats)
    - Dates
    - Spans of text from the passage
    - Lists of numbers/spans
    """
    
    def __init__(self, split="train"):
        print(f"Loading DROP ({split})...")
        # DROP uses "train" and "validation" splits
        if split == "test":
            split = "validation"  # DROP doesn't have test, use validation
        self.data = load_dataset("drop", split=split)
        self.name = "drop"
        print(f"  Loaded {len(self.data)} examples")
    
    def get_sample(self):
        """Returns (question, answer) tuple.
        
        For DROP, we format the question with the passage context.
        """
        idx = random.randint(0, len(self.data) - 1)
        sample = self.data[idx]
        
        # DROP structure: {'passage': str, 'question': str, 'answers_spans': {...}}
        passage = sample['passage']
        question = sample['question']
        
        # Format: passage + question (similar to how models see it)
        formatted_question = f"{passage}\n\nQuestion: {question}"
        
        # Extract answer - DROP has 'answers_spans' with 'spans' (list of answer strings)
        # The answer can be a number, date, or text span
        answer_spans = sample.get('answers_spans', {})
        if 'spans' in answer_spans and len(answer_spans['spans']) > 0:
            # Use first answer (usually there's one primary answer)
            answer = answer_spans['spans'][0]
        else:
            # Fallback: try to find answer in other fields
            answer = sample.get('answer', '')
        
        return formatted_question, answer
    
    def _normalize_number(self, text: str) -> Union[float, None]:
        """Extract and normalize a number from text."""
        # Remove commas, currency symbols, etc.
        text = text.replace(',', '').replace('$', '').strip()
        
        # Try to extract number (handles integers and floats)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            try:
                return float(nums[0])
            except ValueError:
                return None
        return None
    
    def _normalize_date(self, text: str) -> str:
        """Normalize date strings for comparison."""
        # Remove common date prefixes/suffixes
        text = re.sub(r'\b(on|the|of)\b', '', text, flags=re.IGNORECASE)
        text = text.strip()
        return text
    
    def _extract_answer_from_prediction(self, prediction: str) -> str:
        """Extract the answer from model prediction.
        
        DROP answers can be:
        1. Numbers (extract first number)
        2. Dates (extract date-like strings)
        3. Text spans (extract quoted text or final answer)
        """
        # Try to find "Answer:" or "Final Answer:" pattern
        answer_patterns = [
            r'Final Answer[:\s]+(.+?)(?:\.|$|\n)',
            r'Answer[:\s]+(.+?)(?:\.|$|\n)',
            r'####\s*(.+)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: use last sentence or last quoted text
        # Remove quotes and extract
        if '"' in prediction:
            quoted = re.findall(r'"([^"]+)"', prediction)
            if quoted:
                return quoted[-1]
        
        # Last resort: return last non-empty line
        lines = [l.strip() for l in prediction.split('\n') if l.strip()]
        if lines:
            return lines[-1]
        
        return prediction.strip()
    
    def evaluate_correctness(self, prediction: str, ground_truth: str) -> float:
        """
        Evaluate correctness for DROP.
        
        DROP answers can be:
        - Numbers: Compare numerically
        - Dates: Compare normalized date strings
        - Text spans: Exact or substring match
        - Lists: Check if prediction contains any answer from list
        """
        # Extract answer from prediction
        pred_answer = self._extract_answer_from_prediction(prediction)
        pred_answer = pred_answer.strip()
        ground_truth = ground_truth.strip()
        
        # Try numerical comparison first
        pred_num = self._normalize_number(pred_answer)
        truth_num = self._normalize_number(ground_truth)
        
        if pred_num is not None and truth_num is not None:
            # Both are numbers - compare numerically
            return 1.0 if abs(pred_num - truth_num) < 1e-3 else 0.0
        
        # Try date comparison
        pred_date = self._normalize_date(pred_answer)
        truth_date = self._normalize_date(ground_truth)
        if pred_date and truth_date:
            # Normalize and compare dates
            if pred_date.lower() == truth_date.lower():
                return 1.0
        
        # Text span comparison (case-insensitive, allow substring match)
        pred_lower = pred_answer.lower()
        truth_lower = ground_truth.lower()
        
        # Exact match
        if pred_lower == truth_lower:
            return 1.0
        
        # Substring match (prediction contains ground truth or vice versa)
        if truth_lower in pred_lower or pred_lower in truth_lower:
            return 1.0
        
        # Check if ground truth is a list (multiple valid answers)
        # DROP sometimes has multiple valid answer formats
        if ',' in ground_truth or ';' in ground_truth:
            truth_parts = [t.strip() for t in re.split(r'[,;]', ground_truth)]
            for part in truth_parts:
                if part.lower() in pred_lower or pred_lower in part.lower():
                    return 1.0
        
        return 0.0

