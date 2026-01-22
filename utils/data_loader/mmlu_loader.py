from datasets import load_dataset
import random
import re
from typing import List, Optional, Dict
from .base_loader import BaseDataset


# MMLU subject categories mapping (verified against cais/mmlu on HuggingFace)
MMLU_SUBJECTS = {
    # Math subjects
    "math": [
        "abstract_algebra",
        "college_mathematics",
        "elementary_mathematics",
        "high_school_mathematics",
        "high_school_statistics",
    ],
    # Physics subjects
    "physics": [
        "college_physics",
        "conceptual_physics",
        "high_school_physics",
    ],
    # Biology subjects
    "bio": [
        "anatomy",
        "college_biology",
        "high_school_biology",
    ],
    # Chemistry subjects
    "chemistry": [
        "college_chemistry",
        "high_school_chemistry",
    ],
    # Computer Science
    "cs": [
        "college_computer_science",
        "high_school_computer_science",
        "computer_security",
        "machine_learning",
    ],
  
    "other": [
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "econometrics",
        "electrical_engineering",
        "formal_logic",
        "global_facts",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ],
}

# Reverse mapping: subject -> category
SUBJECT_TO_CATEGORY = {}
for category, subjects in MMLU_SUBJECTS.items():
    for subject in subjects:
        SUBJECT_TO_CATEGORY[subject] = category


class MMLUDataset(BaseDataset):
    """
    MMLU (Massive Multitask Language Understanding) dataset loader.
    
    Supports:
    - Multiple subjects (math, physics, bio, chemistry, etc.)
    - Test/validation/dev splits (MMLU has no 'train' split, uses 'dev' for few-shot)
    - Subject tags for filtering and analysis
    
    Args:
        split: "test", "validation", or "dev" (MMLU has no 'train' - use 'dev' for few-shot examples)
        subjects: List of subject names (e.g., ["high_school_mathematics", "college_physics"])
                  or category names (e.g., ["math", "physics"])
                  or None to load all subjects
        categories: List of category names (e.g., ["math", "bio"]) to load all subjects in those categories
        seed: Random seed for shuffling
    """
    
    def __init__(self, split="test", subjects: Optional[List[str]] = None, 
                 categories: Optional[List[str]] = None, seed: int = 42):
        # MMLU doesn't have 'train' split - map to 'dev' (few-shot examples)
        if split == "train":
            print(f"Warning: MMLU has no 'train' split. Using 'dev' for few-shot examples instead.")
            split = "dev"
        print(f"Loading MMLU ({split})...")
        
        # Determine which subjects to load
        subjects_to_load = self._resolve_subjects(subjects, categories)
        
        if not subjects_to_load:
            raise ValueError("No subjects specified. Provide 'subjects' or 'categories' parameter.")
        
        print(f"  Loading {len(subjects_to_load)} subjects: {', '.join(subjects_to_load[:5])}{'...' if len(subjects_to_load) > 5 else ''}")
        
        # Load data from all specified subjects
        self.data = []
        self.subject_tags = []  # Track which subject each sample belongs to
        
        for subject in subjects_to_load:
            try:
                # MMLU on HuggingFace: "cais/mmlu" with subject as config
                # Train split has few-shot examples, test has full evaluation set
                dataset = load_dataset("cais/mmlu", subject, split=split)
                
                # Add subject tag to each sample
                for sample in dataset:
                    self.data.append(sample)
                    self.subject_tags.append(subject)
                
                print(f"    Loaded {len(dataset)} samples from {subject}")
            except Exception as e:
                print(f"    Warning: Could not load subject '{subject}': {e}")
                continue
        
        if not self.data:
            raise ValueError(f"No data loaded. Check subject names and split '{split}'.")
        
        # Shuffle data
        combined = list(zip(self.data, self.subject_tags))
        random.Random(seed).shuffle(combined)
        self.data, self.subject_tags = zip(*combined)
        self.data = list(self.data)
        self.subject_tags = list(self.subject_tags)
        
        # Set dataset name based on categories/subjects
        if categories:
            self.name = f"mmlu_{'_'.join(categories)}"
        elif subjects and len(subjects) <= 3:
            self.name = f"mmlu_{'_'.join([s.replace('_', '')[:8] for s in subjects])}"
        else:
            self.name = "mmlu"
        
        print(f"  Total samples: {len(self.data)}")
        
        # Print category distribution
        category_counts = {}
        for tag in self.subject_tags:
            cat = SUBJECT_TO_CATEGORY.get(tag, "other")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f"  Category distribution: {category_counts}")
    
    def _resolve_subjects(self, subjects: Optional[List[str]], 
                         categories: Optional[List[str]]) -> List[str]:
        """
        Resolve subjects list from subjects and/or categories.
        
        If categories are provided (e.g., ["math", "physics"]), loads ALL subjects
        from ALL specified categories (union/combined). For example:
        - categories=["math", "physics"] loads all math subjects + all physics subjects
        - This is what "mmlu_math_physics" does: combines both categories
        """
        subjects_to_load = []
        
        if categories:
            # Expand categories to subjects (combines all subjects from all categories)
            for category in categories:
                if category in MMLU_SUBJECTS:
                    subjects_to_load.extend(MMLU_SUBJECTS[category])
                else:
                    print(f"  Warning: Unknown category '{category}'. Available: {list(MMLU_SUBJECTS.keys())}")
        
        if subjects:
            # Add explicit subjects
            for subject in subjects:
                if subject not in subjects_to_load:
                    subjects_to_load.append(subject)
        
        return subjects_to_load
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample by index. Returns dict with question, choices, answer, subject."""
        sample = self.data[idx]
        return {
            'question': sample['question'],
            'choices': sample['choices'],
            'answer': sample['answer'],
            'subject': self.subject_tags[idx],
            'category': SUBJECT_TO_CATEGORY.get(self.subject_tags[idx], "other")
        }
    
    def get_question(self, idx):
        """Get formatted question string at index (with options)."""
        sample = self.data[idx]
        question_text = sample['question']
        choices = sample['choices']
        options_text = "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(choices)])
        return f"{question_text}\n\nOptions:\n{options_text}"
    
    def get_answer(self, idx):
        """Get correct answer string at index."""
        sample = self.data[idx]
        return sample['choices'][sample['answer']]
    
    def get_sample(self):
        """Returns (question, answer) tuple."""
        idx = random.randint(0, len(self.data) - 1)
        sample = self.data[idx]
        
        # MMLU structure: {'question': str, 'choices': List[str], 'answer': int (0-3)}
        question_text = sample['question']
        choices = sample['choices']  # List of 4 options
        answer_idx = sample['answer']  # 0, 1, 2, or 3
        
        # Format question with options (similar to MedQA)
        options_text = "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(choices)])
        question = f"{question_text}\n\nOptions:\n{options_text}"
        
        # Get correct answer text
        answer = choices[answer_idx]
        
        return question, answer
    
    def get_sample_with_tag(self):
        """Returns (question, answer, subject_tag, category_tag) tuple."""
        idx = random.randint(0, len(self.data) - 1)
        sample = self.data[idx]
        subject_tag = self.subject_tags[idx]
        category_tag = SUBJECT_TO_CATEGORY.get(subject_tag, "other")
        
        question_text = sample['question']
        choices = sample['choices']
        answer_idx = sample['answer']
        
        options_text = "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(choices)])
        question = f"{question_text}\n\nOptions:\n{options_text}"
        answer = choices[answer_idx]
        
        return question, answer, subject_tag, category_tag
    
    def evaluate_correctness(self, prediction: str, ground_truth: str) -> float:
        """
        Evaluate correctness for MMLU (multiple choice).
        Handles letter answers (A, B, C, D) and full text matching.
        """
        pred = prediction.strip().upper()
        truth = ground_truth.strip().upper()
        
        # Extract letter answers (A, B, C, D)
        pred_letter = re.search(r'\b([A-D])\b', pred)
        truth_letter = re.search(r'\b([A-D])\b', truth)
        
        if pred_letter and truth_letter:
            return 1.0 if pred_letter.group(1) == truth_letter.group(1) else 0.0
        
        # Match full text (case-insensitive, normalized)
        pred_normalized = re.sub(r'[^\w\s]', '', pred.lower())
        truth_normalized = re.sub(r'[^\w\s]', '', truth.lower())
        
        if truth_normalized in pred_normalized or pred_normalized in truth_normalized:
            return 1.0
        
        return 0.0
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of samples by category."""
        category_counts = {}
        for tag in self.subject_tags:
            cat = SUBJECT_TO_CATEGORY.get(tag, "other")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        return category_counts
    
    def get_subject_distribution(self) -> Dict[str, int]:
        """Get distribution of samples by subject."""
        subject_counts = {}
        for tag in self.subject_tags:
            subject_counts[tag] = subject_counts.get(tag, 0) + 1
        return subject_counts
