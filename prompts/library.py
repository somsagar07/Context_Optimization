"""
Prompt atoms library for RL-based prompt optimization.

Each agent (Reasoner, Verifier, Answerer) has its own set of prompt atoms
that can be selected sequentially to build a customized system prompt.
"""
import os
import json

# Reasoner prompt atoms - for step-by-step problem solving
REASONER_ATOMS = {
    0: None,  # DONE - stop selecting
    1: "Think through this step-by-step before giving your answer.",
    2: "Break this problem into smaller sub-problems and solve each one.",
    3: "Identify the key variables, entities, or parameters involved in this problem.",
    4: "Recall relevant general knowledge or principles that apply to this domain.",
    5: "Think step-by-step, explaining the logical connection between each step.",
    6: "Show all intermediate calculations explicitly.",
}

# Verifier prompt atoms - for checking and validating
VERIFIER_ATOMS = {
    0: None,  # DONE
    1: "Check if the reasoning strictly follows the constraints given in the prompt.",
    2: "Review the logic for any jumps in reasoning or hallucinations.",
    3: "Substitute the answer back into the original problem to verify.",
    4: "Check if the answer is reasonable given the context.",
    5: "Look for edge cases or counter-examples that might invalidate the solution.",
}

# Answerer prompt atoms - for formatting final output
ANSWERER_ATOMS = {
    0: None,  # DONE
    1: "Synthesize the gathered information into a concise, direct response.",
    2: "If there are conflicting results, select the one with the strongest supporting evidence.",
    3: "Format the final answer clearly, separating the result from the explanation.",
    4: "Briefly explain how you got the answer.",
}

# # Router: For classifying questions (Simple vs Complex, Math vs Retrieval)
# ROUTER_ATOMS = {
#     0: None,
#     1: "Classify this problem based on the tools required to solve it.",
#     2: "Determine if this is a single-step or multi-step reasoning problem.",
#     3: "Assess if this question requires external knowledge or pure logic.",
#     4: "Identify the domain of the question (e.g., Arithmetic, History, Coding).",
# }

# # Orchestrator: For breaking down complex tasks
# ORCHESTRATOR_ATOMS = {
#     0: None,
#     1: "Break this task into independent sub-tasks that can be solved in parallel.",
#     2: "Identify the logical dependencies between steps.",
#     3: "Create a step-by-step plan assigning specific tools to each step.",
#     4: "Extract the core variables and identify missing information.",
# }

# # Aggregator: For synthesizing multiple results (Voting/Workers)
# AGGREGATOR_ATOMS = {
#     0: None,
#     1: "Compare the provided solutions and identify the most consistent answer.",
#     2: "Resolve conflicts between the different results by checking assumptions.",
#     3: "Synthesize the partial results into a coherent final response.",
#     4: "If results disagree, analyze which derivation method was more robust.",
# }


# Combined reference
PROMPT_ATOMS = {
    "reasoner": REASONER_ATOMS,
    "verifier": VERIFIER_ATOMS,
    "answerer": ANSWERER_ATOMS,
    # "router": ROUTER_ATOMS,
    # "orchestrator": ORCHESTRATOR_ATOMS,
    # "aggregator": AGGREGATOR_ATOMS,
}

NUM_ATOMS = {
    "reasoner": len(REASONER_ATOMS),
    "verifier": len(VERIFIER_ATOMS),
    "answerer": len(ANSWERER_ATOMS),
    # "router": len(ROUTER_ATOMS),
    # "orchestrator": len(ORCHESTRATOR_ATOMS),
    # "aggregator": len(AGGREGATOR_ATOMS),
}

# REASONER_DONE = 0
# NUM_REASONER_ATOMS = 7  # 6 prompts + 1 DONE

# VERIFIER_DONE = 0
# NUM_VERIFIER_ATOMS = 6  # 5 prompts + 1 DONE

# ANSWERER_DONE = 0
# NUM_ANSWERER_ATOMS = 5  # 4 prompts + 1 DONE

def _get_atoms_path(dataset_name: str) -> str:
    """Returns the path where generated atoms for this dataset should be stored."""
    # Saves to: prompts/generated/<dataset_name>/atoms.json
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "generated", dataset_name, "atoms.json")

def load_or_create_atoms(dataset_name: str, worker=None):
    """
    Core function to manage dynamic atoms.
    1. Checks if atoms exist for this dataset.
    2. If yes, loads them into the global dictionaries.
    3. If no and worker provided, generates them, saves them, then loads.
    """
    file_path = _get_atoms_path(dataset_name)
    
    # 1. Try to load existing
    if os.path.exists(file_path):
        print(f"[Prompts] Found existing atoms for '{dataset_name}'. Loading...")
        _load_from_file(file_path)
        return

    # 2. If missing and we have a worker, Generate
    if worker:
        print(f"[Prompts] No atoms found for '{dataset_name}'. Generating new ones...")
        try:
            # Import here to avoid circular dependency at module level
            from .generator import AtomGenerator
            
            generator = AtomGenerator(worker)
            new_atoms = generator.generate_all_atoms(dataset_name)
            
            # Save to disk
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(new_atoms, f, indent=4)
            
            print(f"[Prompts] Saved generated atoms to {file_path}")
            
            # Load the newly created file
            _load_from_file(file_path)
            
        except Exception as e:
            print(f"[Prompts] Error generating atoms: {e}")
    else:
        print(f"[Prompts] Warning: No atoms found for {dataset_name} and no worker provided to generate them.")


def _load_from_file(file_path: str):
    """Helper to merge loaded JSON atoms into the global dicts."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Helper to append dicts safely (using string keys from JSON as int)
        def append_atoms(target_dict, source_dict):
            """
            Appends atoms from source_dict to target_dict using the next available
            sequential integer keys. Ignores the keys in source_dict (e.g., "100").
            """
            if not source_dict:
                return

            # Find the next available index (e.g., if max is 6, start at 7)
            # Default to 1 if dict only has 0:None or is empty (though 0 is usually present)
            current_max = max(target_dict.keys()) if target_dict else 0
            next_idx = current_max + 1
            
            # Sort source items by their original key to ensure deterministic order
            sorted_items = sorted(source_dict.items(), key=lambda x: int(x[0]))
            
            count = 0
            for _, text in sorted_items:
                target_dict[next_idx] = text
                next_idx += 1
                count += 1
            return count

        if "reasoner" in data:
            append_atoms(REASONER_ATOMS, data["reasoner"])
        if "verifier" in data:
            append_atoms(VERIFIER_ATOMS, data["verifier"])
        if "answerer" in data:
            append_atoms(ANSWERER_ATOMS, data["answerer"])
        
        # Update the counts
        refresh_counts()
        
        print(f"[Prompts] Atoms loaded.  Reasoner: {len(REASONER_ATOMS)}, Verifier: {len(VERIFIER_ATOMS)}, Answerer: {len(ANSWERER_ATOMS)}")

    except Exception as e:
        print(f"[Prompts] Error reading atoms file: {e}")


def refresh_counts():
    """Updates the NUM_ATOMS global based on current dictionaries."""
    global NUM_ATOMS
    NUM_ATOMS = {k: len(v) for k, v in PROMPT_ATOMS.items()}


def build_prompt_suffix(agent_type: str, selected_indices: list) -> str:
    """
    Build a prompt suffix from selected atom indices.
    
    Args:
        agent_type: "reasoner", "verifier", or "answerer"
        selected_indices: List of atom indices (excluding 0/DONE)
    
    Returns:
        Combined prompt text, or empty string if no prompts selected
    """
    if not selected_indices:
        return ""
    
    atoms = PROMPT_ATOMS.get(agent_type, {})
    fragments = []
    
    for idx in selected_indices:
        text = atoms.get(idx)
        if text:
            fragments.append(text)
    
    if not fragments:
        return ""
    
    return " ".join(fragments)


def get_atom_text(agent_type: str, index: int) -> str:
    """Get the text for a specific atom."""
    atoms = PROMPT_ATOMS.get(agent_type, {})
    return atoms.get(index, "")