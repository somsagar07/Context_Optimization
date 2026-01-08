"""
Generate atoms for all API model-dataset combinations.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
atom_gen_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, atom_gen_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

import generator_wrapper
from config import API_MODELS_TO_TEST, DATASETS, API_RATE_LIMIT_DELAY, RESULTS_DIR
from models import get_model_metadata


def generate_atoms_for_model_dataset(
    model_id: str,
    dataset_name: str,
    results_dir: str = RESULTS_DIR,
    delay: float = None
) -> Dict:
    """
    Generate atoms for a specific model-dataset combination.
    
    Args:
        model_id: OpenRouter model ID
        dataset_name: Dataset name
        results_dir: Results directory
        delay: Delay between API calls
        
    Returns:
        Dictionary with generated atoms and metadata
    """
    if delay is None:
        delay = API_RATE_LIMIT_DELAY
    
    print(f"\n{'='*80}")
    print(f"Generating atoms: {model_id} × {dataset_name}")
    print(f"{'='*80}")
    
    try:
        # Create generator
        generator = generator_wrapper.create_atom_generator(model_id)
        
        # Generate atoms
        atoms = generator.generate_atoms(dataset_name, delay=delay)
        
        # Get metadata
        metadata = generator.get_metadata()
        metadata["dataset"] = dataset_name
        metadata["generation_time"] = time.time()
        
        # Save results
        model_dir = model_id.replace("/", "-")  # Sanitize model ID for directory name
        output_dir = Path(results_dir) / "generated_atoms" / model_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save atoms
        atoms_file = output_dir / "atoms.json"
        with open(atoms_file, 'w') as f:
            json.dump(atoms, f, indent=2)
        
        # Save metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved to {atoms_file}")
        
        return {
            "success": True,
            "model_id": model_id,
            "dataset": dataset_name,
            "atoms": atoms,
            "metadata": metadata,
            "output_dir": str(output_dir),
        }
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {
            "success": False,
            "model_id": model_id,
            "dataset": dataset_name,
            "error": str(e),
        }


def generate_all_atoms(
    models: list = None,
    datasets: list = None,
    results_dir: str = RESULTS_DIR,
    delay: float = None
) -> Dict:
    """
    Generate atoms for all model-dataset combinations.
    
    Args:
        models: List of model IDs (uses config default if None)
        datasets: List of dataset names (uses config default if None)
        results_dir: Results directory
        delay: Delay between API calls
        
    Returns:
        Dictionary with all results
    """
    if models is None:
        models = API_MODELS_TO_TEST
    if datasets is None:
        datasets = DATASETS
    if delay is None:
        delay = API_RATE_LIMIT_DELAY
    
    print(f"\n{'='*80}")
    print(f"ATOM GENERATION ABLATION STUDY")
    print(f"{'='*80}")
    print(f"Models: {len(models)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Total combinations: {len(models) * len(datasets)}")
    print(f"{'='*80}")
    
    all_results = {}
    total_combinations = len(models) * len(datasets)
    completed = 0
    
    for model_id in models:
        all_results[model_id] = {}
        
        for dataset_name in datasets:
            result = generate_atoms_for_model_dataset(
                model_id,
                dataset_name,
                results_dir=results_dir,
                delay=delay
            )
            
            all_results[model_id][dataset_name] = result
            completed += 1
            
            print(f"\nProgress: {completed}/{total_combinations} combinations completed")
            
            # Add delay between combinations
            if completed < total_combinations:
                time.sleep(delay)
    
    # Save summary
    summary_file = Path(results_dir) / "generated_atoms" / "generation_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Generation complete!")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    # Run generation
    results = generate_all_atoms()
    
    # Print summary
    successful = sum(
        1 for model_results in results.values()
        for result in model_results.values()
        if result.get("success", False)
    )
    total = sum(len(model_results) for model_results in results.values())
    
    print(f"\nSummary: {successful}/{total} combinations successful")

