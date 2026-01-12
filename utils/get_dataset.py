from .data_loader import (
    GSM8kDataset,
    HotPotQADataset,
    GAIADataset,
    MedQADataset,
    AIME25Dataset,
    Tau2Dataset
)

def get_dataset_loader(name: str, is_eval: bool = False, domain: str = None):
    """
    Factory function to get the right dataset loader.
    
    Args:
        name: Dataset name (e.g., "tau2_airline", "tau2_retail", "tau2_telecom", "gsm8k", etc.)
        is_eval: If True, loads evaluation dataset
        domain: For tau2 datasets, specify domain (airline, retail, telecom)
                If name starts with "tau2_", domain is extracted from name
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
        # GAIADataset uses rl_split parameter (string, not boolean)
        rl_split = "validation" if is_eval else "train"
        return GAIADataset(rl_split=rl_split)
    elif name == "medqa":
        # openlifescienceai/medqa uses "dev" for validation split
        split = "dev" if is_eval else "train"
        return MedQADataset(split=split)
    elif name == "aime25":
        # AIME25 only has "test" split, we split it internally
        split = "test" if is_eval else "train"
        return AIME25Dataset(split=split)
    else:
        raise ValueError(f"Unknown dataset: {name}. Available: gsm8k, hotpotqa, gaia, medqa, aime25, tau2_airline, tau2_retail, tau2_telecom")