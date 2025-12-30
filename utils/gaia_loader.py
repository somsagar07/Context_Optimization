from datasets import load_dataset
from huggingface_hub import snapshot_download
import os
import string
import warnings
import random
import re
from .base_loader import BaseDataset

class GAIADataset(BaseDataset):
    """GAIA dataset for evaluating agentic reasoning."""
    
    def __init__(self, year: str = "2023", level: str = "all", split: str = "validation", rl_split: str = "train", seed: int = 42):
        assert year in ["2023"], "Year must be '2023'."
        assert level in ["level1", "level2", "level3", "all"], "Level must be 'level1', 'level2', 'level3', or 'all'."
        assert split in ["validation", "test"], "Split must be 'validation' or 'test'."
        
        # Download the dataset snapshot.  This function caches the content locally.
        self.data_dir = snapshot_download(
            repo_id="gaia-benchmark/GAIA", repo_type="dataset"
        )
        config = f"{year}_{level}"
        full_dataset = load_dataset(self.data_dir, config, split=split)
        full_dataset = full_dataset.shuffle(seed=seed)
        
        total_len = len(full_dataset)
        split_idx = 65
        
        if rl_split == "train":
            print(f"GAIA: Using RL Training split (0 to {split_idx})")
            self.data = full_dataset.select(range(0, split_idx))
        elif rl_split == "eval":
            print(f"GAIA: Using RL Test split ({split_idx} to {total_len})")
            self.data = full_dataset.select(range(split_idx, total_len))
        else:
            self.data = full_dataset
    
    def get_sample(self):
        idx = random.randint(0, len(self.data) - 1)
        sample = self.data[idx]
        
        rel_path = sample.get('file_path', '')
        
        full_path = None
        if rel_path:
            full_path = os.path.join(self.data_dir, rel_path)
            # print(f"Loading GAIA sample from {full_path}.")
            
        return sample['Question'], sample['Final answer'], full_path
    
    def evaluate_correctness(self, model_answer: str, ground_truth: str) -> float:
        def _normalize_number_str(number_str: str) -> float:
            # we replace these common units and commas to allow
            # conversion to float
            for char in ["$", "%", ","]:
                number_str = number_str.replace(char, "")
            try:
                return float(number_str)
            except ValueError:
                # print(f"String {number_str} cannot be normalized to number str.")
                return float("inf")
        
        def _split_string(
            s: str,
            char_list: list[str] = [",", ";"],
        ) -> list[str]:
            pattern = f"[{''.join(char_list)}]"
            return re.split(pattern, s)
        
        def _is_float(element: any) -> bool:
            try:
                float(element)
                return True
            except ValueError:
                return False
        
        def _normalize_str(input_str, remove_punct=True) -> str:
            """
            Normalize a string by:
            - Removing all white spaces
            - Optionally removing punctuation (if remove_punct is True)
            - Converting to lowercase
            Parameters:
            - input_str: str, the string to normalize
            - remove_punct: bool, whether to remove punctuation (default: True)
            Returns:
            - str, the normalized string
            """
            # Remove all white spaces. Required e.g for seagull vs. sea gull
            no_spaces = re.sub(r"\s", "", input_str)

            # Remove punctuation, if specified.
            if remove_punct:
                translator = str.maketrans("", "", string.punctuation)
                return no_spaces.lower().translate(translator)
            else:
                return no_spaces.lower()
            
        if model_answer is None:
            model_answer = "None"

        # if gt is a number
        if _is_float(ground_truth):
            # print(f"Evaluating {model_answer} as a number.")
            normalized_answer = _normalize_number_str(model_answer)
            return 1.0 if normalized_answer == float(ground_truth) else 0.0

        # if gt is a list
        elif any(char in ground_truth for char in [",", ";"]):
            # print(f"Evaluating {model_answer} as a comma separated list.")
            # question with the fish: normalization removes punct

            gt_elems = _split_string(ground_truth)
            ma_elems = _split_string(model_answer)

            # check length is the same
            if len(gt_elems) != len(ma_elems):
                warnings.warn(
                    "Answer lists have different lengths, returning False.", UserWarning
                )
                return 0.0

            # compare each element as float or str
            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if _is_float(gt_elem):
                    normalized_ma_elem = _normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    # we do not remove punct since comparisons can include punct
                    comparisons.append(
                        _normalize_str(ma_elem, remove_punct=False)
                        == _normalize_str(gt_elem, remove_punct=False)
                    )
            return 1.0 if all(comparisons) else 0.0

        # if gt is a str
        else:
            # print(f"Evaluating {model_answer} as a string.")
            return 1.0 if _normalize_str(model_answer) == _normalize_str(ground_truth) else 0.0
