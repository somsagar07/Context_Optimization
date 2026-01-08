"""
Consensus scoring for cross-model evaluation of generated atoms.

Each model evaluates all atoms from all models via OpenRouter API.
"""

import os
import sys
import time
import random
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
atom_gen_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, atom_gen_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

from agents_system import OpenRouterWorker
from config import METRICS_CONFIG, API_RATE_LIMIT_DELAY


class ConsensusScorer:
    """Compute consensus scores by having each model evaluate all atoms."""
    
    def __init__(self, rating_scale: Tuple[int, int] = None):
        """
        Initialize consensus scorer.
        
        Args:
            rating_scale: Rating scale tuple (min, max)
        """
        config = METRICS_CONFIG["consensus"]
        self.rating_scale = rating_scale or config["rating_scale"]
        self.rating_criteria = config["rating_criteria"]
        self.delay = API_RATE_LIMIT_DELAY
    
    def _extract_rating(self, text: str) -> Optional[float]:
        """Extract numeric rating from LLM response."""
        # Look for numbers in the response
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            rating = float(numbers[0])
            # Clamp to rating scale
            min_rating, max_rating = self.rating_scale
            rating = max(min_rating, min(max_rating, rating))
            return rating
        return None
    
    def _create_evaluation_prompt(self, atom: str, dataset: str, role: str, criteria: str) -> str:
        """Create evaluation prompt for a specific atom and criterion."""
        prompt = (
            f"You are evaluating a prompt instruction for a '{role}' agent in the '{dataset}' dataset.\n\n"
            f"Instruction: {atom}\n\n"
            f"Rate this instruction for {criteria} on a scale of {self.rating_scale[0]}-{self.rating_scale[1]}.\n"
            f"Consider:\n"
            f"- Overall quality: Is it well-formed and useful?\n"
            f"- Relevance to dataset: Does it address dataset-specific challenges?\n"
            f"- Usefulness for role: Does it help the {role} agent perform its function?\n\n"
            f"Respond with ONLY a number between {self.rating_scale[0]} and {self.rating_scale[1]}."
        )
        return prompt
    
    def evaluate_atom(
        self,
        atom: str,
        dataset: str,
        role: str,
        evaluator_model: str,
        criteria: str = "overall_quality"
    ) -> float:
        """
        Have a model evaluate a single atom.
        
        Args:
            atom: Atom text to evaluate
            dataset: Dataset name
            role: Role (reasoner/verifier/answerer)
            evaluator_model: OpenRouter model ID to use for evaluation
            criteria: Evaluation criteria
            
        Returns:
            Rating score
        """
        try:
            worker = OpenRouterWorker(model_name=evaluator_model)
            prompt = self._create_evaluation_prompt(atom, dataset, role, criteria)
            response = worker.answer_direct(prompt)
            rating = self._extract_rating(response)
            
            if rating is None:
                # Default to middle of scale if extraction fails
                rating = (self.rating_scale[0] + self.rating_scale[1]) / 2.0
            
            time.sleep(self.delay)
            return float(rating)
            
        except Exception as e:
            print(f"Error evaluating atom with {evaluator_model}: {e}")
            # Return middle of scale on error
            return (self.rating_scale[0] + self.rating_scale[1]) / 2.0
    
    def evaluate_all_atoms(
        self,
        all_atoms: Dict[str, Dict[str, Dict[int, str]]],
        evaluator_model: str
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Have a model evaluate all atoms from all models.
        
        Args:
            all_atoms: Nested dict: {model_id: {dataset: {role: {index: atom_text}}}}
            evaluator_model: OpenRouter model ID to use for evaluation
            
        Returns:
            Dictionary with ratings: {model_id: {dataset: {role: {index: rating}}}}
        """
        print(f"\nEvaluating all atoms using {evaluator_model}...")
        
        results = {}
        total_atoms = 0
        
        # Flatten atoms for random ordering
        atom_list = []
        for model_id, datasets in all_atoms.items():
            for dataset, roles in datasets.items():
                for role, atoms_dict in roles.items():
                    for index, atom_text in atoms_dict.items():
                        atom_list.append({
                            "model_id": model_id,
                            "dataset": dataset,
                            "role": role,
                            "index": index,
                            "atom": atom_text,
                        })
                        total_atoms += 1
        
        # Randomize order
        random.shuffle(atom_list)
        
        print(f"  Evaluating {total_atoms} atoms...")
        
        # Evaluate each atom
        for i, atom_info in enumerate(atom_list):
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i + 1}/{total_atoms}")
            
            rating = self.evaluate_atom(
                atom_info["atom"],
                atom_info["dataset"],
                atom_info["role"],
                evaluator_model,
                criteria="overall_quality"
            )
            
            # Store result
            model_id = atom_info["model_id"]
            dataset = atom_info["dataset"]
            role = atom_info["role"]
            index = atom_info["index"]
            
            if model_id not in results:
                results[model_id] = {}
            if dataset not in results[model_id]:
                results[model_id][dataset] = {}
            if role not in results[model_id][dataset]:
                results[model_id][dataset][role] = {}
            
            results[model_id][dataset][role][index] = rating
        
        print(f"  ✓ Completed evaluation")
        return results
    
    def compute_consensus_metrics(
        self,
        all_evaluations: Dict[str, Dict[str, Dict[str, Dict[str, Dict[int, float]]]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute consensus metrics from all model evaluations.
        
        Args:
            all_evaluations: {evaluator_model: {model_id: {dataset: {role: {index: rating}}}}}
            
        Returns:
            Dictionary with consensus metrics per model-dataset-role
        """
        # Aggregate ratings by atom
        atom_ratings = {}  # {(model_id, dataset, role, index): [ratings]}
        
        for evaluator_model, evaluations in all_evaluations.items():
            for model_id, datasets in evaluations.items():
                for dataset, roles in datasets.items():
                    for role, atoms_dict in roles.items():
                        for index, rating in atoms_dict.items():
                            key = (model_id, dataset, role, index)
                            if key not in atom_ratings:
                                atom_ratings[key] = []
                            atom_ratings[key].append(rating)
        
        # Compute metrics
        results = {}
        
        for (model_id, dataset, role, index), ratings in atom_ratings.items():
            if model_id not in results:
                results[model_id] = {}
            if dataset not in results[model_id]:
                results[model_id][dataset] = {}
            if role not in results[model_id][dataset]:
                results[model_id][dataset][role] = {}
            
            ratings_array = np.array(ratings)
            
            results[model_id][dataset][role][index] = {
                "average_rating": float(np.mean(ratings_array)),
                "std_rating": float(np.std(ratings_array)),
                "agreement_score": float(1.0 / (1.0 + np.std(ratings_array))),  # Inverse of std (higher = more agreement)
                "min_rating": float(np.min(ratings_array)),
                "max_rating": float(np.max(ratings_array)),
                "num_evaluators": len(ratings),
            }
        
        return results
    
    def compute_model_bias(
        self,
        all_evaluations: Dict[str, Dict[str, Dict[str, Dict[str, Dict[int, float]]]]]
    ) -> Dict[str, float]:
        """
        Compute bias: how each model rates its own atoms vs others' atoms.
        
        Args:
            all_evaluations: {evaluator_model: {model_id: {dataset: {role: {index: rating}}}}}
            
        Returns:
            Dictionary with bias scores per model
        """
        bias_scores = {}
        
        for evaluator_model, evaluations in all_evaluations.items():
            self_ratings = []
            other_ratings = []
            
            for model_id, datasets in evaluations.items():
                for dataset, roles in datasets.items():
                    for role, atoms_dict in roles.items():
                        for index, rating in atoms_dict.items():
                            if model_id == evaluator_model:
                                self_ratings.append(rating)
                            else:
                                other_ratings.append(rating)
            
            if self_ratings and other_ratings:
                self_mean = np.mean(self_ratings)
                other_mean = np.mean(other_ratings)
                bias = self_mean - other_mean
                bias_scores[evaluator_model] = float(bias)
            else:
                bias_scores[evaluator_model] = 0.0
        
        return bias_scores

