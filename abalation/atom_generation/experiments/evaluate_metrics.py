"""
Evaluate diversity and quality metrics for generated atoms.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
atom_gen_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, atom_gen_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

from metrics.diversity import DiversityMetrics
from metrics.quality import QualityMetrics
from config import RESULTS_DIR, API_MODELS_TO_TEST, DATASETS
from models import get_model_metadata


def load_generated_atoms(results_dir: str = RESULTS_DIR) -> Dict:
    """
    Load all generated atoms from disk.
    
    Args:
        results_dir: Results directory
        
    Returns:
        Dictionary: {model_id: {dataset: {role: {index: atom_text}}}}
    """
    atoms_dir = Path(results_dir) / "generated_atoms"
    all_atoms = {}
    
    for model_dir in atoms_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name == "generation_summary.json":
            continue
        
        # Convert back from sanitized name by matching with known model IDs
        # Directory names are model_id.replace("/", "-")
        dir_name = model_dir.name
        
        # Try to find matching model ID from config
        model_id = None
        for known_id in API_MODELS_TO_TEST:
            if known_id.replace("/", "-") == dir_name:
                model_id = known_id
                break
        
        # Fallback: try simple conversion (replace first dash with slash)
        if model_id is None:
            if "-" in dir_name:
                parts = dir_name.split("-", 1)
                if len(parts) == 2:
                    model_id = f"{parts[0]}/{parts[1]}"
                else:
                    model_id = dir_name
            else:
                model_id = dir_name
        
        all_atoms[model_id] = {}
        
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            atoms_file = dataset_dir / "atoms.json"
            
            if not atoms_file.exists():
                continue
            
            with open(atoms_file, 'r') as f:
                atoms_data = json.load(f)
            
            all_atoms[model_id][dataset_name] = atoms_data
    
    return all_atoms


def convert_atoms_to_lists(atoms_dict: Dict) -> Dict[str, List[str]]:
    """
    Convert atoms from dict format to list format for metrics.
    
    Args:
        atoms_dict: {role: {index: atom_text}}
        
    Returns:
        {role: [atom_text, ...]}
    """
    result = {}
    for role, atoms in atoms_dict.items():
        result[role] = [text for text in atoms.values() if text]
    return result


def evaluate_metrics_for_model_dataset(
    model_id: str,
    dataset_name: str,
    atoms: Dict,
    results_dir: str = RESULTS_DIR,
    diversity_calculator: DiversityMetrics = None,
    quality_calculator: QualityMetrics = None
) -> Dict:
    """
    Evaluate metrics for a specific model-dataset combination.
    
    Args:
        model_id: Model ID
        dataset_name: Dataset name
        atoms: Atoms dictionary {role: {index: atom_text}}
        results_dir: Results directory
        diversity_calculator: Optional pre-initialized diversity calculator (for reuse)
        quality_calculator: Optional pre-initialized quality calculator (for reuse)
        
    Returns:
        Dictionary with metrics
    """
    # Convert to list format
    atoms_lists = convert_atoms_to_lists(atoms)
    
    # Compute diversity metrics (reuse calculator if provided)
    if diversity_calculator is None:
        diversity_calculator = DiversityMetrics()
    diversity_metrics = diversity_calculator.compute_all_metrics(atoms_lists)
    
    # Compute quality metrics (reuse calculator if provided)
    if quality_calculator is None:
        quality_calculator = QualityMetrics()
    quality_metrics = quality_calculator.compute_all_metrics(atoms_lists, model_id=model_id, dataset_name=dataset_name)
    
    # Combine results
    results = {
        "model_id": model_id,
        "dataset": dataset_name,
        "diversity": diversity_metrics,
        "quality": quality_metrics,
    }
    
    # Save results
    model_dir = model_id.replace("/", "-")
    output_dir = Path(results_dir) / "metrics" / model_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def evaluate_all_metrics(
    results_dir: str = RESULTS_DIR,
    models: List[str] = None,
    datasets: List[str] = None
) -> Dict:
    """
    Evaluate metrics for all model-dataset combinations.
    
    Args:
        results_dir: Results directory
        models: List of model IDs (uses config default if None)
        datasets: List of dataset names (uses config default if None)
        
    Returns:
        Dictionary with all metrics
    """
    if models is None:
        models = API_MODELS_TO_TEST
    if datasets is None:
        datasets = DATASETS
    
    print(f"\n{'='*80}")
    print(f"METRIC EVALUATION")
    print(f"{'='*80}")
    
    # Load all generated atoms
    print("Loading generated atoms...")
    all_atoms = load_generated_atoms(results_dir)
    
    if not all_atoms:
        print("  ✗ No generated atoms found. Run generate_atoms.py first.")
        return {}
    
    print(f"  ✓ Loaded atoms for {len(all_atoms)} models")
    
    # Load existing metrics to preserve them when adding new models
    aggregated_file = Path(results_dir) / "metrics" / "all_metrics.json"
    existing_metrics = {}
    if aggregated_file.exists():
        print("  Loading existing metrics to preserve them...")
        with open(aggregated_file, 'r') as f:
            existing_metrics = json.load(f)
        print(f"  ✓ Found existing metrics for {len(existing_metrics)} models")
    
    # Create reusable calculator instances (workers/embeddings are lazy-loaded and cached)
    print("  Initializing metric calculators (will reuse across all combinations)...")
    diversity_calculator = DiversityMetrics()
    quality_calculator = QualityMetrics()
    
    # Start with existing metrics (will be merged with new ones)
    all_metrics = existing_metrics.copy()
    
    # Build list of valid combinations for progress tracking
    valid_combinations = []
    for model_id in models:
        if model_id not in all_atoms:
            continue
        for dataset_name in datasets:
            if dataset_name in all_atoms[model_id]:
                valid_combinations.append((model_id, dataset_name))
    
    if not valid_combinations:
        print("  ⚠ No valid model-dataset combinations found to evaluate")
        print("  Using existing metrics only")
        return all_metrics
    
    # Progress bar for model-dataset combinations
    with tqdm(total=len(valid_combinations), desc="Evaluating metrics", unit="combination", ncols=100) as pbar:
        for model_id, dataset_name in valid_combinations:
            if model_id not in all_metrics:
                all_metrics[model_id] = {}
            
            # Update progress bar with current model/dataset
            model_short = model_id.split("/")[-1] if "/" in model_id else model_id
            pbar.set_postfix({"model": model_short[:20], "dataset": dataset_name})
            
            atoms = all_atoms[model_id][dataset_name]
            metrics = evaluate_metrics_for_model_dataset(
                model_id,
                dataset_name,
                atoms,
                results_dir=results_dir,
                diversity_calculator=diversity_calculator,
                quality_calculator=quality_calculator
            )
            
            # Merge new metrics (overwrites existing if present)
            all_metrics[model_id][dataset_name] = metrics
            pbar.update(1)
    
    # Save merged aggregated results
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(aggregated_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Count newly evaluated vs existing
    newly_evaluated = len(valid_combinations)
    total_models = len(all_metrics)
    total_combinations = sum(len(datasets) for datasets in all_metrics.values())
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"  Newly evaluated: {newly_evaluated} combinations")
    print(f"  Total in aggregated file: {total_combinations} combinations across {total_models} models")
    print(f"Aggregated metrics saved to: {aggregated_file}")
    print(f"{'='*80}")
    
    return all_metrics


if __name__ == "__main__":
    # Run evaluation
    results = evaluate_all_metrics()
    
    # Print summary
    total = sum(len(model_results) for model_results in results.values())
    print(f"\nSummary: Evaluated metrics for {total} model-dataset combinations")

