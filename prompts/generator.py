import json
import os
import random
import sys
from typing import Dict, List
import re
import codecs

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
    
    def _clean_atom_text(self, text: str) -> str:
        """
        Clean and normalize generated atom text.
        Includes normalization of smart quotes, dashes, and symbols to ASCII.
        """
        if not text:
            return ""
        
        # Step 1: Remove common prefixes first (case-insensitive)
        prefixes_to_remove = [
            r"^Final Answer:\s*",
            r"^Answer:\s*",
            r"^Output:\s*",
            r"^System instruction:\s*",
            r"^Instruction:\s*",
            r"^Prompt:\s*",
            r"^Response:\s*",
        ]
        for prefix_pattern in prefixes_to_remove:
            text = re.sub(prefix_pattern, "", text, flags=re.IGNORECASE)
        
        # Step 2: Decode UTF-8 bytes (Your existing logic)
        def decode_utf8_bytes_in_text(text):
            result = []
            i = 0
            while i < len(text):
                char = text[i]
                code_point = ord(char)
                if 0x80 <= code_point <= 0xFF:
                    decoded = None
                    best_length = 0
                    for length in range(1, min(5, len(text) - i + 1)):
                        if i + length > len(text): break
                        byte_chars = []
                        valid = True
                        for k in range(i, i + length):
                            cp = ord(text[k])
                            if 0x80 <= cp <= 0xFF: byte_chars.append(text[k])
                            else: valid = False; break
                        if valid and len(byte_chars) == length:
                            try:
                                byte_sequence = bytes([ord(c) for c in byte_chars])
                                decoded = byte_sequence.decode('utf-8')
                                best_length = length
                            except (UnicodeDecodeError, ValueError):
                                if best_length == 0: continue
                                else: break
                    if decoded and best_length > 0:
                        result.append(decoded)
                        i += best_length
                    else:
                        result.append(char)
                        i += 1
                else:
                    result.append(char)
                    i += 1
            return ''.join(result)
        
        text = decode_utf8_bytes_in_text(text)

        # Step 2.5: Normalize Smart Punctuation and Symbols to ASCII
        # This fixes \u2019, \u201c, \u2192, etc.
        replacements = {
            # Quotes
            "\u2018": "'",  # Left Single Quote
            "\u2019": "'",  # Right Single Quote
            "\u201A": "'",  # Single Low-9 Quote
            "\u201B": "'",  # Single High-Reversed-9 Quote
            "\u201C": '"',  # Left Double Quote
            "\u201D": '"',  # Right Double Quote
            "\u201E": '"',  # Double Low-9 Quote
            "\u201F": '"',  # Double High-Reversed-9 Quote
            
            # Dashes and Hyphens
            "\u2013": "-",  # En Dash
            "\u2014": "-",  # Em Dash
            "\u2011": "-",  # Non-Breaking Hyphen
            
            # Spaces and Ellipsis
            "\u00A0": " ",  # Non-Breaking Space
            "\u2026": "...", # Ellipsis
            
            # Symbols often found in LLM logic outputs
            "\u2192": "->", # Right Arrow (→)
            "\u2190": "<-", # Left Arrow (←)
        }
        
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)

        # Step 3: Remove surrounding quotes (Modified to handle straight quotes now)
        text = re.sub(r'^["\']+|["\']+$', '', text)
        
        # Step 4: Remove leading/trailing whitespace and normalize internal whitespace
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text

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
            
            # Retry logic for failed or empty generations
            max_retries = 3
            atom_text = None
            for attempt in range(max_retries):
                try:
                    response = self.worker.answer_direct("Generate instruction", prompt_suffix=meta_prompt)
                    
                    if response:
                        atom_text = self._clean_atom_text(response)
                        if atom_text:  # Valid non-empty atom
                            break
                    if attempt < max_retries - 1:
                        print(f"  ⚠ Empty response for {role} atom {i+1}, retrying ({attempt+1}/{max_retries})...")
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  ⚠ Error generating {role} atom {i+1}: {e}, retrying ({attempt+1}/{max_retries})...")
                    else:
                        print(f"  ✗ Failed to generate {role} atom {i+1} after {max_retries} attempts: {e}")
            
            # Start keys at 100 to avoid conflict with default atoms 0-6
            key = 100 + i
            if atom_text:
                atoms[key] = atom_text
            else:
                print(f"  ✗ Skipping {role} atom {i+1} (empty after {max_retries} attempts)")
            
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
                if text and text.strip():  # Only add non-empty atoms
                    reasoner[current_idx] = text
                    current_idx += 1
        
        verifier = {}
        current_idx = 100
        atoms = self.generate_atoms_for_role(dataset_name, "verifier", 5)
        for _, text in atoms.items():
            if text and text.strip():  # Only add non-empty atoms
                verifier[current_idx] = text
                current_idx += 1
        
        answerer = {}
        current_idx = 100
        concepts = [("answerer", 3), ("aggregator", 2)]
        for concept, count in concepts:
            atoms = self.generate_atoms_for_role(dataset_name, concept, count)
            for _, text in atoms.items():
                if text and text.strip():  # Only add non-empty atoms
                    answerer[current_idx] = text
                    current_idx += 1
        
        # Clean all atoms before returning
        def clean_atoms_dict(atoms_dict):
            """Clean all atom texts in a dictionary."""
            cleaned = {}
            for key, value in atoms_dict.items():
                cleaned[key] = self._clean_atom_text(value) if value else value
            return cleaned
        
        return {
            "reasoner": clean_atoms_dict(reasoner),
            "verifier": clean_atoms_dict(verifier),
            "answerer": clean_atoms_dict(answerer),
        }