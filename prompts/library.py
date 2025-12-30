"""
Prompt atoms library for RL-based prompt optimization.

Each agent (Reasoner, Verifier, Answerer) has its own set of prompt atoms
that can be selected sequentially to build a customized system prompt.
"""

# Reasoner prompt atoms - for step-by-step problem solving
REASONER_ATOMS = {
    0: None,  # DONE - stop selecting
    1: "Think through this step-by-step before giving your answer.",
    2: "Break this problem into smaller sub-problems and solve each one.",
    3: "Identify what each number in the problem represents.",
    4: "Set up an equation or formula to model this problem.",
    5: "First estimate the answer, then calculate precisely.",
    6: "Show all intermediate calculations explicitly.",
}
REASONER_DONE = 0
NUM_REASONER_ATOMS = 7  # 6 prompts + 1 DONE

# Verifier prompt atoms - for checking and validating
VERIFIER_ATOMS = {
    0: None,  # DONE
    1: "Check each calculation step for arithmetic errors.",
    2: "Verify the logic and reasoning flow makes sense.",
    3: "Substitute the answer back into the original problem to verify.",
    4: "Check if the answer is reasonable given the context.",
    5: "Look for common mistakes like sign errors or unit mismatches.",
}
VERIFIER_DONE = 0
NUM_VERIFIER_ATOMS = 6  # 5 prompts + 1 DONE

# Answerer prompt atoms - for formatting final output
ANSWERER_ATOMS = {
    0: None,  # DONE
    1: "Be concise and state only the final answer.",
    2: "Format your final answer as: #### <number>",
    3: "Include the units in your final answer.",
    4: "Briefly explain how you got the answer.",
}
ANSWERER_DONE = 0
NUM_ANSWERER_ATOMS = 5  # 4 prompts + 1 DONE

# Combined reference
PROMPT_ATOMS = {
    "reasoner": REASONER_ATOMS,
    "verifier": VERIFIER_ATOMS,
    "answerer": ANSWERER_ATOMS,
}

NUM_ATOMS = {
    "reasoner": NUM_REASONER_ATOMS,
    "verifier": NUM_VERIFIER_ATOMS,
    "answerer": NUM_ANSWERER_ATOMS,
}


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

