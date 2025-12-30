from .gsm8k_loader import GSM8kDataset
from .hotpot_loader import HotPotQADataset
from .gaia_loader import GAIADataset

def get_dataset_loader(name: str):
    """Factory function to get the right dataset loader."""
    if name == "gsm8k":
        return GSM8kDataset()
    elif name == "hotpotqa":
        return HotPotQADataset()
    elif name == "gaia":
        return GAIADataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")