from .gsm8k_loader import GSM8kDataset
from .hotpot_loader import HotPotQADataset
from .gaia_loader import GAIADataset
from .medqa_loader import MedQADataset

def get_dataset_loader(name: str, is_eval: bool = False):
    """Factory function to get the right dataset loader."""
    if name == "gsm8k":
        split = "test" if is_eval else "train"
        return GSM8kDataset(split=split)
    elif name == "hotpotqa":
        split = "validation" if is_eval else "train"
        return HotPotQADataset(split=split)
    elif name == "gaia":
        # GAIADataset uses rl_split parameter (string, not boolean)
        rl_split = "validation" if is_eval else "train"
        return GAIADataset(rl_split=rl_split)
    elif name == "medqa":
        # openlifescienceai/medqa uses "dev" for validation split
        split = "dev" if is_eval else "train"
        return MedQADataset(split=split)
    else:
        raise ValueError(f"Unknown dataset: {name}. Available: gsm8k, hotpotqa, gaia, medqa")