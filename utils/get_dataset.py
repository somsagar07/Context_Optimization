from .gsm8k_loader import GSM8kDataset
from .hotpot_loader import HotPotQADataset
from .gaia_loader import GAIADataset

def get_dataset_loader(name: str, is_eval: bool = False):
    """Factory function to get the right dataset loader."""
    if name == "gsm8k":
        return GSM8kDataset(split=is_eval)
    elif name == "hotpotqa":
        return HotPotQADataset(split=is_eval)
    elif name == "gaia":
        return GAIADataset(rl_split=is_eval)
    else:
        raise ValueError(f"Unknown dataset: {name}")