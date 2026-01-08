"""
Wrapper around AtomGenerator to use OpenRouterWorker for API-based atom generation.
"""

import os
import sys
import time
from typing import Dict, Optional

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

from agents_system import OpenRouterWorker
from prompts.generator import AtomGenerator
from models import get_model_metadata


class APIAtomGenerator:
    """
    Wrapper around AtomGenerator that uses OpenRouterWorker for API-based generation.
    Handles API errors, rate limiting, and metadata tracking.
    """
    
    def __init__(self, model_id: str, api_key: Optional[str] = None):
        """
        Initialize API atom generator.
        
        Args:
            model_id: OpenRouter model ID (e.g., "openai/gpt-4o-mini")
            api_key: OpenRouter API key (optional, uses env var if not provided)
        """
        self.model_id = model_id
        self.metadata = get_model_metadata(model_id)
        
        # Create OpenRouterWorker
        try:
            self.worker = OpenRouterWorker(model_name=model_id, api_key=api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenRouterWorker for {model_id}: {e}")
        
        # Create AtomGenerator with the worker
        self.generator = AtomGenerator(self.worker)
        
        # Track API usage (optional)
        self.api_calls = 0
        self.api_errors = 0
    
    def generate_atoms(self, dataset_name: str, delay: float = 1.5) -> Dict[str, Dict[int, str]]:
        """
        Generate atoms for a dataset with rate limiting.
        
        Args:
            dataset_name: Name of the dataset
            delay: Delay in seconds between API calls
            
        Returns:
            Dictionary with generated atoms per role
        """
        print(f"\nGenerating atoms for {dataset_name} using {self.model_id}...")
        print(f"  Model: {self.metadata['display_name']} ({self.metadata['family']})")
        
        start_time = time.time()
        
        try:
            # Generate atoms (AtomGenerator handles internal API calls)
            atoms = self.generator.generate_all_atoms(dataset_name)
            
            # Add delay after generation to respect rate limits
            time.sleep(delay)
            
            elapsed_time = time.time() - start_time
            
            print(f"  ✓ Generated atoms in {elapsed_time:.2f}s")
            print(f"    Reasoner: {len(atoms.get('reasoner', {}))} atoms")
            print(f"    Verifier: {len(atoms.get('verifier', {}))} atoms")
            print(f"    Answerer: {len(atoms.get('answerer', {}))} atoms")
            
            return atoms
            
        except Exception as e:
            self.api_errors += 1
            print(f"  ✗ Error generating atoms: {e}")
            raise
    
    def get_metadata(self) -> Dict:
        """Get metadata about the model and generation stats."""
        return {
            "model_id": self.model_id,
            "model_metadata": self.metadata,
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
        }


def create_atom_generator(model_id: str, api_key: Optional[str] = None) -> APIAtomGenerator:
    """
    Factory function to create an APIAtomGenerator.
    
    Args:
        model_id: OpenRouter model ID
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        APIAtomGenerator instance
    """
    return APIAtomGenerator(model_id, api_key)

