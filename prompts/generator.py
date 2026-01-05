import json
import os
import random
import sys
from typing import Dict, List

# Ensure we can import from parent directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.get_dataset import get_dataset_loader

class AtomGenerator:
    """
    Generates dataset-specific prompt atoms using an LLMWorker.
    """
    
    def __init__(self, worker):
        self.worker = worker
        
        # Strategies to force the LLM to produce diverse instructions
        self.strategies = {
            "analytical": "Focus on strict logical decomposition and formal math.",
            "creative": "Encourage lateral thinking and novel problem-solving approaches.",
            "pedagogical": "Act like a teacher explaining concepts clearly.",
            "critical": "Focus on error detection and verifying assumptions.",
            "expert_persona": "Adopt the persona of a domain expert (e.g., mathematician, scientist).",
            "constraint_focused": "Emphasize adhering to strict formatting and constraints."
        }

    def _get_dataset_examples(self, dataset_name: str, n: int = 3) -> str:
        """Fetches real examples from the dataset to ground the generation."""
        try:
            # Load training data for analysis
            loader = get_dataset_loader(dataset_name, is_eval=False)
            
            # FIX: Access the underlying Hugging Face dataset if it's wrapped
            # Your loaders (GSM8kDataset, etc.) store the actual data in .data
            if hasattr(loader, 'data'):
                dataset = loader.data
            else:
                dataset = loader

            examples_text = []
            count = 0
            
            # Now we can iterate safely
            for item in dataset:
                if count >= n:
                    break
                
                # 1. Extract Question
                # Handle GSM8k/HotpotQA 'question' vs GAIA 'Question'
                q = item.get('question') or item.get('Question') or item.get('input') or str(item)
                
                # 2. Extract File Context (Specific to GAIA/Multimodal)
                # Your gaia_loader.py checks for 'file_path' and adds a notification.
                # We replicate that here so the prompt generator knows files are involved.
                file_path = item.get('file_path')
                if file_path:
                    q += f"\n[System Notification] File Attachment: {file_path}"

                # 3. Extract Answer
                # Handle 'answer' vs 'Final answer' (GAIA) vs 'ground_truth'
                a = item.get('answer') or item.get('Final answer') or item.get('output') or item.get('ground_truth') or ""
                
                examples_text.append(f"Example {count+1}:\nInput: {q}\nTarget Output: {a}\n")
                count += 1
                
            return "\n".join(examples_text)
        except Exception as e:
            print(f"Warning: Could not load examples for {dataset_name}: {e}")
            return "No examples available."

    def _generate_dataset_summary(self, dataset_name: str, examples: str) -> str:
        """Asks the LLM to analyze what makes this dataset difficult."""
        meta_prompt = (
            f"Analyze the following examples from the '{dataset_name}' dataset:\n\n"
            f"{examples}\n\n"
            "Identify the key reasoning patterns, specific difficulties (e.g., multi-step arithmetic, retrieval), "
            "and what a model needs to do well here. Provide a 2-sentence summary."
        )
        return self.worker.reason("Analyze this dataset", prompt_suffix=meta_prompt)
    
    def generate_atoms_for_role(self, dataset_name: str, role: str, count: int = 5) -> Dict[int, str]:
        """
        Generates specific atoms for a given role (reasoner/verifier/answerer).
        """
        examples = self._get_dataset_examples(dataset_name)
        dataset_summary = self._generate_dataset_summary(dataset_name, examples)
        
        atoms = {}
        
        role_goals = {
            "reasoner": "guide the step-by-step thinking process",
            "verifier": "critique and find errors in reasoning",
            "answerer": "format the final output concisely",
            "router": "analyze the question complexity and select the right approach/agent",
            "orchestrator": "decompose the problem into sub-tasks or parallel threads",
            "aggregator": "resolve conflicts between multiple different answers or sub-task results",
        }
        goal = role_goals.get(role, "solve the task")

        selected_strategies = random.sample(list(self.strategies.items()), min(count, len(self.strategies)))
        
        print(f"Generating {len(selected_strategies)} {role} atoms for {dataset_name}...")

        for i, (strat_name, strat_desc) in enumerate(selected_strategies):
            meta_prompt = (
                f"You are an expert prompt engineer optimizing for the '{dataset_name}' dataset.\n"
                f"Dataset Analysis: {dataset_summary}\n"
                f"Goal: Write a ONE-SENTENCE system instruction for a '{role}' agent to {goal}.\n"
                f"Strategy: {strat_desc}\n"
                f"Real Examples:\n{examples}\n\n"
                f"Output ONLY the instruction sentence. Do not include quotes or prefixes."
            )
            
            atom_text = self.worker.answer_direct("Generate instruction", prompt_suffix=meta_prompt)
            atom_text = atom_text.replace('"', '').replace("System instruction:", "").strip()
            
            # Start keys at 100 to avoid conflict with default atoms 0-6
            key = 100 + i 
            atoms[key] = atom_text
            
        return atoms

    def generate_all_atoms(self, dataset_name: str) -> Dict[str, Dict[int, str]]:
        # return {
        #     "reasoner": self.generate_atoms_for_role(dataset_name, "reasoner"),
        #     "verifier": self.generate_atoms_for_role(dataset_name, "verifier"),
        #     "answerer": self.generate_atoms_for_role(dataset_name, "answerer"),
        #     "router": self.generate_atoms_for_role(dataset_name, "router"),
        #     "orchestrator": self.generate_atoms_for_role(dataset_name, "orchestrator"),
        #     "aggregator": self.generate_atoms_for_role(dataset_name, "aggregator"),
        # }
        
        reasoner = {}
        current_idx = 100
        
        concepts = [("reasoner", 3), ("router", 2), ("orchestrator", 2)]
        for concept, count in concepts:
            atoms = self.generate_atoms_for_role(dataset_name, concept, count)
            for _, text in atoms.items():
                reasoner[current_idx] = text
                current_idx += 1
        
        verifier = {}
        current_idx = 100
        atoms = self.generate_atoms_for_role(dataset_name, "verifier", 5)
        for _, text in atoms.items():
            verifier[current_idx] = text
            current_idx += 1
        
        answerer = {}
        current_idx = 100
        concepts = [("answerer", 3), ("aggregator", 2)]
        for concept, count in concepts:
            atoms = self.generate_atoms_for_role(dataset_name, concept, count)
            for _, text in atoms.items():
                answerer[current_idx] = text
                current_idx += 1
        
        return {
            "reasoner": reasoner,
            "verifier": verifier,
            "answerer": answerer,
        }