"""
Quality metrics for evaluating generated prompt atoms.

Includes coherence, specificity, and clarity metrics.
Uses API models for LLM-based quality evaluations.
"""

import os
import sys
import time
import re
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
atom_gen_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, atom_gen_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

from agents_system import OpenRouterWorker
from agents_system.worker import MetaCLIPEmbedder
from prompts.library import REASONER_ATOMS, VERIFIER_ATOMS, ANSWERER_ATOMS
from config import METRICS_CONFIG, API_RATE_LIMIT_DELAY


class QualityMetrics:
    """Compute quality metrics for generated atoms."""
    
    def __init__(self, evaluator_model: str = None, embedding_model: str = None):
        """
        Initialize quality metrics calculator.
        
        Args:
            evaluator_model: OpenRouter model ID for LLM-based evaluations
            embedding_model: Sentence transformer model for embeddings
        """
        config = METRICS_CONFIG["quality"]
        
        self.evaluator_model = evaluator_model or config["coherence"]["evaluator_model"]
        self.clarity_evaluator_model = config["clarity"]["evaluator_model"]  # Use clarity's evaluator model
        self.embedding_model_name = embedding_model or config["specificity"]["embedding_model"]
        self.rating_scale = config["coherence"]["scale"]
        
        # Lazy load components
        self._evaluator_worker = None
        self._clarity_evaluator_worker = None
        self._embedder = None
        
        # Base atoms for specificity comparison
        self.base_atoms = {
            "reasoner": [v for v in REASONER_ATOMS.values() if v is not None],
            "verifier": [v for v in VERIFIER_ATOMS.values() if v is not None],
            "answerer": [v for v in ANSWERER_ATOMS.values() if v is not None],
        }
    
    @property
    def evaluator_worker(self):
        """Lazy load evaluator worker for coherence."""
        if self._evaluator_worker is None:
            print(f"Initializing evaluator worker: {self.evaluator_model}...")
            self._evaluator_worker = OpenRouterWorker(model_name=self.evaluator_model)
        return self._evaluator_worker
    
    @property
    def clarity_evaluator_worker(self):
        """Lazy load evaluator worker for clarity (may be different from coherence)."""
        if self._clarity_evaluator_worker is None:
            if self.clarity_evaluator_model == self.evaluator_model:
                # Reuse same worker if models are the same
                self._clarity_evaluator_worker = self.evaluator_worker
            else:
                print(f"Initializing clarity evaluator worker: {self.clarity_evaluator_model}...")
                self._clarity_evaluator_worker = OpenRouterWorker(model_name=self.clarity_evaluator_model)
        return self._clarity_evaluator_worker
    
    @property
    def embedder(self):
        """Lazy load embedding model (MetaCLIP or sentence transformer)."""
        if self._embedder is None:
            if self.embedding_model_name == "metaclip":
                print(f"Loading MetaCLIP-H14 embedder...")
                self._embedder = MetaCLIPEmbedder(target_dim=None)  # Use native 1024D
            else:
                print(f"Loading sentence transformer: {self.embedding_model_name}...")
                self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder
    
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
    
    def compute_coherence(self, atom: str, delay: float = None) -> float:
        """
        Compute coherence score using API model evaluation.
        
        Args:
            atom: Atom text to evaluate
            delay: Delay after API call (uses config default if None)
            
        Returns:
            Coherence score (1-10)
        """
        if delay is None:
            delay = API_RATE_LIMIT_DELAY
        
        prompt = (
            f"Rate this instruction for coherence and well-formedness on a scale of {self.rating_scale[0]}-{self.rating_scale[1]}:\n\n"
            f"{atom}\n\n"
            f"Respond with ONLY a number between {self.rating_scale[0]} and {self.rating_scale[1]}."
        )
        
        try:
            response = self.evaluator_worker.answer_direct(prompt)
            rating = self._extract_rating(response)
            
            if rating is None:
                # Default to middle of scale if extraction fails
                rating = (self.rating_scale[0] + self.rating_scale[1]) / 2.0
            
            time.sleep(delay)
            return float(rating)
            
        except Exception as e:
            print(f"Error computing coherence: {e}")
            # Return middle of scale on error
            return (self.rating_scale[0] + self.rating_scale[1]) / 2.0
    
    def compute_specificity(self, atom: str, role: str) -> float:
        """
        Compute specificity score: difference from base atoms.
        
        Higher score = more dataset-specific (less similar to generic base atoms).
        
        Args:
            atom: Atom text to evaluate
            role: Role (reasoner/verifier/answerer)
            
        Returns:
            Specificity score (0-1, higher is better)
        """
        if role not in self.base_atoms or not self.base_atoms[role]:
            return 0.0
        
        # Get embeddings (handle MetaCLIP vs sentence transformers)
        if self.embedding_model_name == "metaclip":
            atom_embedding = np.array([self.embedder.embed(atom)])
            base_embeddings = np.array([self.embedder.embed(base_atom) for base_atom in self.base_atoms[role]])
        else:
            atom_embedding = self.embedder.encode([atom], convert_to_numpy=True)
            base_embeddings = self.embedder.encode(self.base_atoms[role], convert_to_numpy=True)
        
        # Compute similarity to base atoms
        similarities = cosine_similarity(atom_embedding, base_embeddings)[0]
        max_similarity = float(np.max(similarities))
        
        # Specificity = 1 - similarity (higher difference = more specific)
        specificity = 1.0 - max_similarity
        
        return float(specificity)
    
    def compute_clarity(self, atom: str, delay: float = None) -> float:
        """
        Compute clarity score using API model evaluation.
        
        Args:
            atom: Atom text to evaluate
            delay: Delay after API call (uses config default if None)
            
        Returns:
            Clarity score (1-10)
        """
        if delay is None:
            delay = API_RATE_LIMIT_DELAY
        
        prompt = (
            f"Rate this instruction for clarity and actionability on a scale of {self.rating_scale[0]}-{self.rating_scale[1]}:\n\n"
            f"{atom}\n\n"
            f"Respond with ONLY a number between {self.rating_scale[0]} and {self.rating_scale[1]}."
        )
        
        try:
            # Use clarity evaluator worker (cached/lazy-loaded)
            response = self.clarity_evaluator_worker.answer_direct(prompt)
            rating = self._extract_rating(response)
            
            if rating is None:
                # Default to middle of scale if extraction fails
                rating = (self.rating_scale[0] + self.rating_scale[1]) / 2.0
            
            time.sleep(delay)
            return float(rating)
            
        except Exception as e:
            print(f"Error computing clarity: {e}")
            # Return middle of scale on error
            return (self.rating_scale[0] + self.rating_scale[1]) / 2.0
    
    def compute_all_metrics(self, atoms: Dict[str, List[str]], roles: Dict[str, str] = None, model_id: str = None, dataset_name: str = None) -> Dict[str, Dict[str, float]]:
        """
        Compute all quality metrics for atoms organized by role.
        
        Args:
            atoms: Dictionary mapping role (reasoner/verifier/answerer) to list of atom texts
            roles: Optional mapping from atom index to role (if atoms are not organized by role)
            model_id: Optional model ID for progress bar display
            dataset_name: Optional dataset name for progress bar display
            
        Returns:
            Dictionary with metrics per role and overall
        """
        results = {}
        
        # Collect all scores for overall computation (avoid duplicate computation)
        all_coherence = []
        all_specificity = []
        all_clarity = []
        
        # Calculate total atoms for progress bar
        total_atoms = sum(len(atom_list) for atom_list in atoms.values())
        
        # Create progress bar description
        desc_parts = ["Quality metrics"]
        if model_id:
            model_short = model_id.split("/")[-1] if "/" in model_id else model_id
            desc_parts.append(model_short[:15])
        if dataset_name:
            desc_parts.append(dataset_name)
        desc = " | ".join(desc_parts)
        
        # Progress bar for atom evaluation
        pbar = tqdm(total=total_atoms, desc=desc, leave=False, unit="atom", ncols=100)
        
        # Compute metrics per role
        for role, atom_list in atoms.items():
            if not atom_list:
                results[role] = {
                    "coherence": 0.0,
                    "specificity": 0.0,
                    "clarity": 0.0,
                }
                continue
            
            # Compute metrics for each atom (store for overall calculation)
            coherence_scores = []
            specificity_scores = []
            clarity_scores = []
            
            for atom in atom_list:
                coherence = self.compute_coherence(atom)
                specificity = self.compute_specificity(atom, role)
                clarity = self.compute_clarity(atom)
                
                coherence_scores.append(coherence)
                specificity_scores.append(specificity)
                clarity_scores.append(clarity)
                
                # Store for overall calculation
                all_coherence.append(coherence)
                all_specificity.append(specificity)
                all_clarity.append(clarity)
                
                # Update progress bar
                pbar.update(1)
            
            results[role] = {
                "coherence": float(np.mean(coherence_scores)),
                "specificity": float(np.mean(specificity_scores)),
                "clarity": float(np.mean(clarity_scores)),
            }
        
        pbar.close()
        
        # Compute overall average from collected scores (no duplicate computation)
        if all_coherence:
            results["overall"] = {
                "coherence": float(np.mean(all_coherence)),
                "specificity": float(np.mean(all_specificity)),
                "clarity": float(np.mean(all_clarity)),
            }
        else:
            results["overall"] = {
                "coherence": 0.0,
                "specificity": 0.0,
                "clarity": 0.0,
            }
        
        return results

