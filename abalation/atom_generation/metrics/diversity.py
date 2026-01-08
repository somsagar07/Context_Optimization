"""
Diversity metrics for evaluating generated prompt atoms.

Includes uniqueness, strategy coverage, and semantic diversity metrics.
"""

import os
import sys
import numpy as np
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
atom_gen_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, atom_gen_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

from agents_system.worker import MetaCLIPEmbedder
from sentence_transformers import SentenceTransformer

from config import METRICS_CONFIG


class DiversityMetrics:
    """Compute diversity metrics for generated atoms."""
    
    def __init__(self, embedding_model: str = None):
        """
        Initialize diversity metrics calculator.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
        """
        config = METRICS_CONFIG["diversity"]
        self.embedding_model_name = embedding_model or config["uniqueness"]["embedding_model"]
        self.strategies = config["strategy_coverage"]["strategies"]
        # Use embedding model from semantic_diversity config if specified, otherwise use uniqueness model
        semantic_config = config.get("semantic_diversity", {})
        self.semantic_embedding_model = semantic_config.get("embedding_model", self.embedding_model_name)
        self.n_clusters = semantic_config.get("n_clusters", 5)
        self.random_state = semantic_config.get("random_state", 42)
        
        # Lazy load embedding model
        self._embedder = None
    
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
    
    def compute_uniqueness(self, atoms: List[str]) -> float:
        """
        Compute uniqueness score: 1 - mean(pairwise_cosine_similarity)
        
        Higher score = more unique atoms.
        
        Args:
            atoms: List of atom text strings
            
        Returns:
            Uniqueness score (0-1, higher is better)
        """
        if len(atoms) < 2:
            return 0.0
        
        # Get embeddings (handle MetaCLIP vs sentence transformers)
        if self.embedding_model_name == "metaclip":
            embeddings = np.array([self.embedder.embed(atom) for atom in atoms])
        else:
            embeddings = self.embedder.encode(atoms, convert_to_numpy=True)
        
        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal)
        n = len(atoms)
        upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]
        
        # Uniqueness = 1 - mean similarity
        mean_similarity = np.mean(upper_triangle)
        uniqueness = 1.0 - mean_similarity
        
        return float(uniqueness)
    
    def compute_strategy_coverage(self, atoms: List[str]) -> float:
        """
        Compute strategy coverage: unique_strategies / total_strategies
        
        Maps atoms to strategy categories and measures coverage.
        
        Args:
            atoms: List of atom text strings
            
        Returns:
            Strategy coverage score (0-1, higher is better)
        """
        if not atoms:
            return 0.0
        
        # Simple keyword-based strategy detection
        # This could be improved with LLM-based classification
        strategy_keywords = {
            "analytical": ["analyze", "decompose", "break down", "logical", "systematic"],
            "creative": ["creative", "novel", "lateral", "alternative", "innovative"],
            "pedagogical": ["explain", "teach", "clarify", "demonstrate", "show"],
            "critical": ["verify", "check", "validate", "review", "critique"],
            "expert_persona": ["expert", "specialist", "professional", "domain"],
            "constraint_focused": ["format", "constraint", "strict", "adhere", "follow"],
        }
        
        detected_strategies = set()
        
        for atom in atoms:
            atom_lower = atom.lower()
            for strategy, keywords in strategy_keywords.items():
                if any(keyword in atom_lower for keyword in keywords):
                    detected_strategies.add(strategy)
                    break
        
        coverage = len(detected_strategies) / len(self.strategies)
        return float(coverage)
    
    def compute_semantic_diversity(self, atoms: List[str]) -> float:
        """
        Compute semantic diversity using clustering silhouette score.
        
        Higher score = more diverse semantic clusters.
        
        Args:
            atoms: List of atom text strings
            
        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        if len(atoms) < self.n_clusters + 1:
            return 0.0
        
        # Get embeddings (use semantic embedding model if different)
        if self.semantic_embedding_model == "metaclip":
            if self._embedder is None or not isinstance(self._embedder, MetaCLIPEmbedder):
                embedder = MetaCLIPEmbedder(target_dim=None)
            else:
                embedder = self._embedder
            embeddings = np.array([embedder.embed(atom) for atom in atoms])
        else:
            embeddings = self.embedder.encode(atoms, convert_to_numpy=True)
        
        # Cluster using K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Compute silhouette score
        silhouette = silhouette_score(embeddings, cluster_labels)
        
        return float(silhouette)
    
    def compute_all_metrics(self, atoms: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Compute all diversity metrics for atoms organized by role.
        
        Args:
            atoms: Dictionary mapping role (reasoner/verifier/answerer) to list of atom texts
            
        Returns:
            Dictionary with metrics per role
        """
        results = {}
        
        for role, atom_list in atoms.items():
            if not atom_list:
                results[role] = {
                    "uniqueness": 0.0,
                    "strategy_coverage": 0.0,
                    "semantic_diversity": 0.0,
                }
                continue
            
            results[role] = {
                "uniqueness": self.compute_uniqueness(atom_list),
                "strategy_coverage": self.compute_strategy_coverage(atom_list),
                "semantic_diversity": self.compute_semantic_diversity(atom_list),
            }
        
        # Compute overall average
        all_atoms = []
        for atom_list in atoms.values():
            all_atoms.extend(atom_list)
        
        if all_atoms:
            results["overall"] = {
                "uniqueness": self.compute_uniqueness(all_atoms),
                "strategy_coverage": self.compute_strategy_coverage(all_atoms),
                "semantic_diversity": self.compute_semantic_diversity(all_atoms),
            }
        else:
            results["overall"] = {
                "uniqueness": 0.0,
                "strategy_coverage": 0.0,
                "semantic_diversity": 0.0,
            }
        
        return results

