#!/usr/bin/env python3
"""
Precompute MetaCLIP embeddings for all datasets.
Run once before RL training to avoid GPU overhead during training.

Usage:
    python scripts/precompute_embeddings.py --datasets hotpotqa gsm8k medqa
    python scripts/precompute_embeddings.py --all
    python scripts/precompute_embeddings.py --datasets hotpotqa --force  # Recompute
"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from huggingface_hub import snapshot_download


# Default directory to store precomputed embeddings
DEFAULT_EMBEDDINGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "embeddings_cache"
)


def _format_medqa_question(sample):
    """Format MedQA question with options (matches MedQADataset.get_sample)."""
    data = sample["data"]
    question = data["Question"]
    options = data["Options"]
    options_text = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
    return f"{question}\n\nOptions:\n{options_text}"


def _make_gaia_extractor(data_dir: str):
    """Create GAIA question extractor that includes file paths (matches GAIADataset.get_sample)."""
    import os
    def extract_fn(sample):
        question = sample["Question"]
        rel_path = sample.get('file_path', '')
        if rel_path:
            full_path = os.path.join(data_dir, rel_path)
            question += f"\n\n[System Notification]\nFile Attachment: {full_path}\nYou can use your tools to read or process this file."
        return question
    return extract_fn


def _make_tau2_extractor(domain: str):
    """Create Tau2 question extractor (matches Tau2Dataset.get_sample)."""
    def extract_fn(task):
        # Extract user_goal from normalized task structure
        user_goal = task.get('user_goal', 'No explicit goal found.')
        question = (
            f"Customer interaction in {domain.capitalize()} domain.\n"
            f"User's request: {user_goal}"
        )
        return question
    return extract_fn


def _normalize_tau2_task(raw_task):
    """Normalize tau2 task structure (matches Tau2Dataset._normalize_task)."""
    scenario = raw_task.get("user_scenario", {})
    instructions = scenario.get("instructions", {})
    
    goal = instructions.get("reason_for_call")
    if not goal:
        goal = (
            raw_task.get("description") or 
            raw_task.get("goal") or 
            "No explicit goal found."
        )
    
    return {"user_goal": goal, "task_id": raw_task.get("id", "Unknown")}


def compute_question_hash(question: str) -> str:
    """Compute a short hash for a question (for deduplication and lookup)."""
    return hashlib.md5(question.encode()).hexdigest()[:16]


def load_dataset_with_config(dataset_name: str, split: str):
    """Load dataset based on name and split, handling special cases."""
    
    if dataset_name == "hotpotqa":
        data = load_dataset("hotpot_qa", "fullwiki", split=split)
        extract_fn = lambda x: x["question"]
        
    elif dataset_name == "gsm8k":
        data = load_dataset("gsm8k", "main", split=split)
        extract_fn = lambda x: x["question"]
        
    elif dataset_name == "medqa":
        data = load_dataset("openlifescienceai/medqa", split=split)
        extract_fn = _format_medqa_question
        
    elif dataset_name == "aime25":
        # AIME25 only has "test" split, we split it internally
        full_data = load_dataset("math-ai/aime25", split="test")
        if split == "train":
            data = full_data.select(range(0, 20))
        else:  # test/validation
            data = full_data.select(range(20, len(full_data)))
        extract_fn = lambda x: x["problem"]
        
    elif dataset_name == "gaia":
        # GAIA uses snapshot_download
        # Include file paths to match GAIADataset.get_sample() format
        data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")
        data = load_dataset(data_dir, "2023_all", split=split)
        extract_fn = _make_gaia_extractor(data_dir)
    
    elif dataset_name == "drop":
        # DROP uses "train" and "validation" splits
        # Map "test" to "validation" for consistency
        if split == "test":
            split = "validation"
        data = load_dataset("drop", split=split)
        # Format: passage + question (matches DROPDataset.get_sample)
        extract_fn = lambda x: f"{x['passage']}\n\nQuestion: {x['question']}"
    
    elif dataset_name.startswith("mmlu"):
        # MMLU datasets: use the MMLUDataset loader
        from utils.get_dataset import get_dataset_loader
        from utils.data_loader.mmlu_loader import MMLU_SUBJECTS
        
        # Parse categories from name if provided
        categories = None
        if "_" in dataset_name:
            name_parts = dataset_name.split("_")[1:]  # Skip "mmlu"
            categories = [part for part in name_parts if part in MMLU_SUBJECTS.keys()]
        
        # Use the dataset loader to get MMLU dataset
        mmlu_dataset = get_dataset_loader(dataset_name, is_eval=(split != "train"), categories=categories)
        
        # Extract questions from MMLU dataset
        def extract_mmlu_question(idx):
            sample = mmlu_dataset.data[idx]
            question_text = sample['question']
            choices = sample['choices']
            options_text = "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(choices)])
            return f"{question_text}\n\nOptions:\n{options_text}"
        
        # Create a list-like interface for compatibility
        class MMLUDataWrapper:
            def __init__(self, dataset):
                self.dataset = dataset
                self._data = dataset.data
            
            def __len__(self):
                return len(self._data)
            
            def __getitem__(self, idx):
                return {"question": extract_mmlu_question(idx)}
        
        data = MMLUDataWrapper(mmlu_dataset)
        extract_fn = lambda x: x["question"]
    
    elif dataset_name.startswith("tau2_"):
        # Tau2 datasets: tau2_airline, tau2_retail, tau2_telecom
        domain = dataset_name.split("_", 1)[1]
        
        # Load tau2 tasks from local cache or download
        import json
        import requests
        
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache", "tau2")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_path = os.path.join(cache_dir, f"{domain}_tasks.json")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                raw_tasks = json.load(f)
        else:
            # Download from GitHub
            url = f"https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/{domain}/tasks.json"
            print(f"Downloading Tau2 {domain} from {url}...")
            response = requests.get(url)
            response.raise_for_status()
            raw_tasks = response.json()
            with open(cache_path, 'w') as f:
                json.dump(raw_tasks, f)
        
        # Normalize tasks and create a list-like structure
        data = [_normalize_tau2_task(t) for t in raw_tasks]
        extract_fn = _make_tau2_extractor(domain)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return data, extract_fn


# Dataset configurations
DATASET_CONFIGS = {
    "hotpotqa": {
        "splits": ["train", "validation"],
    },
    "gsm8k": {
        "splits": ["train", "test"],
    },
    "medqa": {
        "splits": ["train", "dev"],
    },
    "aime25": {
        "splits": ["train", "test"],
    },
    "gaia": {
        "splits": ["validation"],  # Main split used
    },
    "drop": {
        "splits": ["train", "validation"],
    },
    "tau2_airline": {
        "splits": ["all"],  # Tau2 doesn't have train/test splits
    },
    "tau2_retail": {
        "splits": ["all"],
    },
    "tau2_telecom": {
        "splits": ["all"],
    },
    # MMLU category presets (MMLU has no train split, uses dev/test)
    "mmlu_math": {
        "splits": ["test"],
    },
    "mmlu_physics": {
        "splits": ["test"],
    },
    "mmlu_bio": {
        "splits": ["test"],
    },
    "mmlu_chemistry": {
        "splits": ["test"],
    },
    "mmlu_cs": {
        "splits": ["test"],
    },
}


def precompute_dataset(
    dataset_name: str,
    embedder,
    output_dir: str,
    force: bool = False
):
    """
    Precompute embeddings for a single dataset.
    
    Saves:
        - {dataset_name}_{split}_embeddings.npz: Embeddings array and hashes
    """
    # Get config - for MMLU, create config dynamically if not in DATASET_CONFIGS
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
    elif dataset_name.startswith("mmlu"):
        # MMLU datasets: use test split (MMLU has no train, dev is for few-shot)
        config = {"splits": ["test"]}
    else:
        print(f"Unknown dataset: {dataset_name}. Skipping.")
        return
    os.makedirs(output_dir, exist_ok=True)
    
    for split in config["splits"]:
        output_path = os.path.join(output_dir, f"{dataset_name}_{split}_embeddings.npz")
        
        # Skip if already computed (unless force)
        if os.path.exists(output_path) and not force:
            print(f"✓ {dataset_name}/{split} already computed. Use --force to recompute.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} ({split})...")
        print(f"{'='*60}")
        
        # Load dataset
        try:
            data, extract_fn = load_dataset_with_config(dataset_name, split)
        except Exception as e:
            print(f"Failed to load {dataset_name}/{split}: {e}")
            continue
        
        print(f"Loaded {len(data)} samples from {dataset_name}/{split}")
        
        # Extract questions and compute embeddings
        questions = []
        embeddings = []
        question_hashes = []
        seen_hashes = set()
        skipped_duplicates = 0
        
        for sample in tqdm(data, desc=f"Embedding {dataset_name}/{split}", unit="samples"):
            try:
                question = extract_fn(sample)
                q_hash = compute_question_hash(question)
                
                # Skip duplicates
                if q_hash in seen_hashes:
                    skipped_duplicates += 1
                    continue
                seen_hashes.add(q_hash)
                
                # Compute embedding
                embedding = embedder.embed(question)
                
                questions.append(question)
                embeddings.append(embedding)
                question_hashes.append(q_hash)
                
            except Exception as e:
                tqdm.write(f"Warning: Failed to embed sample: {e}")
                continue
        
        if not embeddings:
            print(f"No embeddings computed for {dataset_name}/{split}. Skipping save.")
            continue
        
        # Save embeddings
        embeddings_array = np.array(embeddings, dtype=np.float32)
        np.savez_compressed(
            output_path,
            embeddings=embeddings_array,
            hashes=np.array(question_hashes, dtype=object),
            questions=np.array(questions, dtype=object)  # For debugging
        )
        
        print(f"\n✓ Saved {len(embeddings)} embeddings to {output_path}")
        print(f"  Skipped duplicates: {skipped_duplicates}")
        print(f"  Embedding shape: {embeddings_array.shape}")
        print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def validate_embedding_dataset(name):
    """Validate dataset name for embedding precomputation."""
    # Standard datasets in DATASET_CONFIGS
    if name in DATASET_CONFIGS:
        return name
    
    # Allow any mmlu_* pattern (for custom combinations like mmlu_math_physics)
    if name.startswith("mmlu"):
        from utils.data_loader.mmlu_loader import MMLU_SUBJECTS
        if name == "mmlu":
            return name
        # Validate categories
        parts = name.split("_")[1:]
        valid_categories = set(MMLU_SUBJECTS.keys())
        for part in parts:
            if part not in valid_categories:
                raise argparse.ArgumentTypeError(
                    f"Invalid MMLU category '{part}' in '{name}'. "
                    f"Valid: {sorted(valid_categories)}"
                )
        return name
    
    raise argparse.ArgumentTypeError(
        f"Unknown dataset: '{name}'. "
        f"Valid: {list(DATASET_CONFIGS.keys())} or any 'mmlu_*' combination"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Precompute MetaCLIP embeddings for datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/precompute_embeddings.py --datasets hotpotqa gsm8k
    python scripts/precompute_embeddings.py --datasets mmlu_math mmlu_bio
    python scripts/precompute_embeddings.py --datasets mmlu_math_physics  # Combined categories
    python scripts/precompute_embeddings.py --all
    python scripts/precompute_embeddings.py --datasets hotpotqa --force
        """
    )
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        type=validate_embedding_dataset,
        help=f"Datasets to precompute. Standard: {list(DATASET_CONFIGS.keys())}. "
             "Also supports 'mmlu_<category>' combinations (e.g., mmlu_math_physics)."
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Precompute all standard datasets (including mmlu_math, mmlu_physics, etc.)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_EMBEDDINGS_DIR,
        help=f"Directory to store embeddings (default: {DEFAULT_EMBEDDINGS_DIR})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if files exist"
    )
    parser.add_argument(
        "--target-dim",
        type=int,
        default=None,
        help="Target embedding dimension (default: native 1024D)"
    )
    
    args = parser.parse_args()
    
    if not args.datasets and not args.all:
        parser.error("Must specify --datasets or --all")
    
    datasets = list(DATASET_CONFIGS.keys()) if args.all else args.datasets
    
    print(f"{'='*60}")
    print("Precompute MetaCLIP Embeddings")
    print(f"{'='*60}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Force recompute: {args.force}")
    print(f"{'='*60}")
    
    # Initialize embedder once
    print("\nInitializing MetaCLIP-H14 embedder...")
    from agents_system.worker import MetaCLIPEmbedder
    embedder = MetaCLIPEmbedder(target_dim=args.target_dim, use_precomputed=False)
    # Force initialization to load the model
    embedder._init_embedder()
    print(f"Embedder ready. Dimension: {embedder.get_dimension()}")
    
    # Process each dataset
    for dataset_name in datasets:
        precompute_dataset(
            dataset_name=dataset_name,
            embedder=embedder,
            output_dir=args.output_dir,
            force=args.force
        )
    
    print(f"\n{'='*60}")
    print("✓ Done! Precomputed embeddings saved to:")
    print(f"  {args.output_dir}")
    print("\nTo use during training, embeddings will be auto-loaded if this directory exists.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

