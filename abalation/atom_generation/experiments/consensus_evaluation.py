"""
Consensus evaluation: Each model evaluates all atoms from all models.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
atom_gen_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, atom_gen_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

from metrics.consensus import ConsensusScorer
from config import RESULTS_DIR, API_MODELS_TO_TEST, DATASETS


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
        
        model_id = model_dir.name.replace("-", "/")
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


def run_consensus_evaluation(
    evaluator_models: List[str] = None,
    results_dir: str = RESULTS_DIR
) -> Dict:
    """
    Run consensus evaluation: each model evaluates all atoms.
    
    Args:
        evaluator_models: List of model IDs to use as evaluators (uses config default if None)
        results_dir: Results directory
        
    Returns:
        Dictionary with all evaluations
    """
    if evaluator_models is None:
        evaluator_models = API_MODELS_TO_TEST
    
    print(f"\n{'='*80}")
    print(f"CONSENSUS EVALUATION")
    print(f"{'='*80}")
    print(f"Evaluator models: {len(evaluator_models)}")
    
    # Load all generated atoms
    print("\nLoading generated atoms...")
    all_atoms = load_generated_atoms(results_dir)
    
    if not all_atoms:
        print("  ✗ No generated atoms found. Run generate_atoms.py first.")
        return {}
    
    print(f"  ✓ Loaded atoms for {len(all_atoms)} models")
    
    # Initialize consensus scorer
    scorer = ConsensusScorer()
    
    # Run evaluation for each evaluator model
    all_evaluations = {}
    
    for evaluator_model in evaluator_models:
        print(f"\n{'='*80}")
        print(f"Evaluator: {evaluator_model}")
        print(f"{'='*80}")
        
        try:
            # Evaluate all atoms
            evaluations = scorer.evaluate_all_atoms(all_atoms, evaluator_model)
            all_evaluations[evaluator_model] = evaluations
            
            # Save individual evaluator results
            evaluator_dir = evaluator_model.replace("/", "-")
            output_dir = Path(results_dir) / "consensus_scores" / evaluator_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            evaluations_file = output_dir / "evaluations.json"
            with open(evaluations_file, 'w') as f:
                json.dump(evaluations, f, indent=2)
            
            print(f"  ✓ Saved to {evaluations_file}")
            
        except Exception as e:
            print(f"  ✗ Error with {evaluator_model}: {e}")
            continue
    
    # Compute consensus metrics
    print(f"\n{'='*80}")
    print(f"Computing consensus metrics...")
    print(f"{'='*80}")
    
    consensus_metrics = scorer.compute_consensus_metrics(all_evaluations)
    model_bias = scorer.compute_model_bias(all_evaluations)
    
    # Save consensus metrics
    consensus_file = Path(results_dir) / "consensus_scores" / "aggregated_consensus.json"
    consensus_file.parent.mkdir(parents=True, exist_ok=True)
    
    consensus_data = {
        "evaluations": all_evaluations,
        "consensus_metrics": consensus_metrics,
        "model_bias": model_bias,
    }
    
    with open(consensus_file, 'w') as f:
        json.dump(consensus_data, f, indent=2)
    
    print(f"  ✓ Saved to {consensus_file}")
    
    print(f"\n{'='*80}")
    print(f"Consensus evaluation complete!")
    print(f"{'='*80}")
    
    return consensus_data


if __name__ == "__main__":
    # Run consensus evaluation
    results = run_consensus_evaluation()
    
    # Print summary
    if results:
        num_evaluators = len(results.get("evaluations", {}))
        print(f"\nSummary: {num_evaluators} models completed consensus evaluation")

