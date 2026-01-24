"""
Dataset loading and validation utilities.
"""
import argparse
from typing import Optional, List
from .data_loader import (
    GSM8kDataset,
    HotPotQADataset,
    GAIADataset,
    MedQADataset,
    AIME25Dataset,
    # Tau2Dataset,
    MMLUDataset
)
from .data_loader.mmlu_loader import MMLU_SUBJECTS


# Standard datasets (exact match required)
STANDARD_DATASETS = [
    "gsm8k", "hotpotqa", "gaia", "medqa", "aime25", 
    "tau2_airline", "tau2_retail", "tau2_telecom"
]


def validate_dataset_name(dataset_name):
    """
    Validate dataset name for argparse. Allows any MMLU combination (mmlu_*) or exact matches for other datasets.
    
    Can be used as argparse type validator:
        parser.add_argument("--dataset", type=validate_dataset_name, ...)
    
    Args:
        dataset_name: The dataset name to validate
        
    Returns:
        dataset_name if valid
        
    Raises:
        argparse.ArgumentTypeError: If dataset name is invalid
        
    Examples:
        - "gsm8k" -> valid
        - "mmlu_math" -> valid
        - "mmlu_math_physics_chemistry" -> valid (any combination)
        - "mmlu_invalid" -> raises ArgumentTypeError
    """
    if dataset_name is None:
        return dataset_name
    
    # Standard datasets (exact match required)
    if dataset_name in STANDARD_DATASETS:
        return dataset_name
    
    # MMLU datasets: allow any combination starting with "mmlu"
    if dataset_name.startswith("mmlu"):
        # Validate that categories after "mmlu_" are valid
        if dataset_name == "mmlu":
            return dataset_name  # "mmlu" means all categories
        
        # Check if categories are valid
        name_parts = dataset_name.split("_")[1:]  # Skip "mmlu"
        valid_categories = set(MMLU_SUBJECTS.keys())
        
        for part in name_parts:
            if part not in valid_categories:
                raise argparse.ArgumentTypeError(
                    f"Invalid MMLU category '{part}' in '{dataset_name}'. "
                    f"Valid categories: {sorted(valid_categories)}"
                )
        return dataset_name
    
    # Unknown dataset
    raise argparse.ArgumentTypeError(
        f"Unknown dataset: '{dataset_name}'. "
        f"Valid datasets: {STANDARD_DATASETS} or any 'mmlu_*' combination"
    )


def get_dataset_help_text():
    """
    Get help text for --dataset argument.
    
    Args:
        None
    Returns:
        Help text string for argparse
    """
    standard = "gsm8k, hotpotqa, gaia, medqa, aime25"
    
    return (
        f"Dataset name. Standard datasets: {standard}. "
        "For MMLU, use 'mmlu' (all) or 'mmlu_<category1>_<category2>...' "
        "(e.g., 'mmlu_math', 'mmlu_math_physics', 'mmlu_math_physics_chemistry'). "
        "Valid MMLU categories: math, physics, bio, chemistry, cs, other"
    )


def get_dataset_loader(name: str, is_eval: bool = False, domain: str = None,
                      subjects: Optional[List[str]] = None,
                      categories: Optional[List[str]] = None):
    """
    Factory function to get the right dataset loader.
    
    Args:
        name: Dataset name (e.g., "tau2_airline", "tau2_retail", "tau2_telecom", "gsm8k", etc.)
        is_eval: If True, loads evaluation dataset
        domain: For tau2 datasets, specify domain (airline, retail, telecom)
                If name starts with "tau2_", domain is extracted from name
        subjects: For MMLU, specify subjects list
        categories: For MMLU, specify categories list (e.g., ["math", "physics"])
    """
    # Handle tau2 datasets
    if name.startswith("tau2_"):
        if domain is None:
            # Extract domain from name (e.g., "tau2_airline" -> "airline")
            domain = name.split("_", 1)[1] if "_" in name else "retail"
        split = "test" if is_eval else "train"
        return Tau2Dataset(split=split, domain=domain)
    
    # Handle other datasets
    if name == "gsm8k":
        split = "test" if is_eval else "train"
        return GSM8kDataset(split=split)
    elif name == "hotpotqa":
        split = "validation" if is_eval else "train"
        return HotPotQADataset(split=split)
    elif name == "gaia":
        # # GAIADataset uses rl_split parameter (string, not boolean)
        # rl_split = "validation" if is_eval else "train"
        # GAIADataset uses rl_split parameter: "train" (0-64) or "eval" (65-165)
        rl_split = "eval" if is_eval else "train"
        return GAIADataset(rl_split=rl_split)
    elif name == "medqa":
        # openlifescienceai/medqa has train/dev/test splits
        # Use "test" for final evaluation, "train" for training
        split = "test" if is_eval else "train"
        return MedQADataset(split=split)
    elif name == "aime25":
        # AIME25 only has "test" split, we split it internally
        split = "test" if is_eval else "train"
        return AIME25Dataset(split=split)
    elif name.startswith("mmlu"):
        # MMLU datasets: 
        # - "mmlu" = all subjects (all categories)
        # - "mmlu_math" = math category only
        # - "mmlu_math_physics" = math AND physics categories (combined)
        split = "test" if is_eval else "train"
        
        # Parse categories from name if provided (e.g., "mmlu_math" or "mmlu_math_physics")
        if name == "mmlu":
            # "mmlu" means all categories (all subjects)
            categories = list(MMLU_SUBJECTS.keys())
        elif categories is None and "_" in name:
            name_parts = name.split("_")[1:]  # Skip "mmlu"
            categories = [part for part in name_parts if part in MMLU_SUBJECTS.keys()]
        
        return MMLUDataset(split=split, subjects=subjects, categories=categories)
    else:
        raise ValueError(f"Unknown dataset: {name}. Available: gsm8k, hotpotqa, gaia, medqa, aime25, tau2_airline, tau2_retail, tau2_telecom, mmlu, mmlu_math, mmlu_physics, mmlu_bio, etc.")