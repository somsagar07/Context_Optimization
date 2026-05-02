"""
Atom Generator V2 (default).

Improvements over v1 (prompts/generator.py):
- Uses anthropic/claude-opus-4-7 by default (configured at the train.py layer).
- Generates descriptive, multi-sentence atoms (technique + when-to-apply + concrete action).
- Caches the dataset summary to disk so it is generated once and reused across roles.
- Pulls more grounding examples (n=5) and tells the LLM the examples cover the full
  task — instructing it to use only the parts relevant to the role being generated.
- Passes the running list of already-generated atoms back into the prompt to discourage
  near-duplicates within and across roles.
- Larger atom pools: reasoner=10, verifier=7, answerer=7.
- Saves atoms to prompts/generated_v2/<dataset>/atoms.json (alongside summary.txt).
"""
import json
import os
import random
from typing import Dict, List, Optional

from .generator import AtomGenerator


class AtomGeneratorV2(AtomGenerator):
    """V2 generator. Subclasses v1 and overrides generation behavior."""

    VERSION = "v2"

    def __init__(self, worker, summary_cache_path: Optional[str] = None):
        super().__init__(worker)
        self.summary_cache_path = summary_cache_path

        # Extend strategies so we can produce up to 10 distinct atoms per role
        # without forcing duplicates from random.sample.
        self.strategies = {
            **self.strategies,  # original 6 from v1
            "formal_proof": "Apply rigorous, theorem-style justification to every step.",
            "iterative_refinement": "Refine the solution through multiple targeted passes.",
            "counterfactual": "Explicitly consider what would happen if assumptions were violated, and probe edge cases.",
            "comparative": "Generate and compare multiple solution paths before committing to one.",
        }

    # Phrases that indicate the LLM leaked the dataset/benchmark identity into the
    # atom text. The atom is appended verbatim to the agent's system prompt, so
    # references like "for MedQA vignettes" / "USMLE-style stem" make atoms read
    # as meta-commentary instead of standing instructions.
    _LEAK_BLOCKLIST = (
        "this dataset", "this benchmark", "the dataset", "the benchmark",
        "usmle", "vignette", "vignettes",
        "gsm8k", "hotpotqa", "hotpot qa", "medqa", "med qa",
        "aime", "gaia", "drop ", "mmlu",
    )

    def _dataset_name_leak(self, text: str, dataset_name: str) -> str:
        """
        Returns a short reason string if the atom text mentions the dataset name
        or a known benchmark identifier; otherwise returns "".
        """
        if not text:
            return ""
        lower = text.lower()
        # Always block the literal dataset_name (handles dynamic mmlu_<subject>).
        if dataset_name and dataset_name.lower() in lower:
            return f"contains '{dataset_name}'"
        for phrase in self._LEAK_BLOCKLIST:
            if phrase in lower:
                return f"contains '{phrase.strip()}'"
        return ""

    @staticmethod
    def _looks_truncated(text: str) -> bool:
        """Heuristic: atom was cut off by the token budget mid-sentence."""
        if not text:
            return True
        stripped = text.rstrip()
        # Final char should be sentence-terminating punctuation.
        return stripped[-1] not in ".!?\""

    def _generate_dataset_summary(self, dataset_name: str, examples: str) -> str:
        """
        Generate (or load cached) a richer dataset analysis used to ground every atom.
        """
        if self.summary_cache_path and os.path.exists(self.summary_cache_path):
            try:
                with open(self.summary_cache_path, "r") as f:
                    cached = f.read().strip()
                    if cached:
                        print(f"[Prompts v2] Loaded cached dataset summary from {self.summary_cache_path}")
                        return cached
            except Exception as e:
                print(f"[Prompts v2] Could not read cached summary, regenerating: {e}")

        meta_prompt = (
            f"Analyze the following examples from the '{dataset_name}' dataset:\n\n"
            f"{examples}\n\n"
            "Provide a thorough analysis (around 6-10 sentences, multiple paragraphs are fine) covering:\n"
            "(1) the specific reasoning patterns the task requires,\n"
            "(2) the discriminating features that distinguish correct answers from distractors,\n"
            "(3) the typical failure modes an LLM exhibits on this task,\n"
            "(4) what distinguishes an expert solver from a naive one,\n"
            "(5) any format or output constraints the final answer must follow.\n"
            "Be concrete and dataset-specific. Cite the kind of inputs / outputs / pitfalls you see in the examples. "
            "This summary will ground all subsequent atom generation, so do not be terse."
        )
        # Summary is grounding context for ~24 downstream atom generations,
        # so we give it room to be thorough.
        summary_token_budget = 2048
        summary = self.worker.reason(
            "Analyze this dataset",
            prompt_suffix=meta_prompt,
            tokens=summary_token_budget,
        ) or ""
        summary = summary.strip()

        if self.summary_cache_path and summary:
            try:
                os.makedirs(os.path.dirname(self.summary_cache_path), exist_ok=True)
                with open(self.summary_cache_path, "w") as f:
                    f.write(summary)
                print(f"[Prompts v2] Cached dataset summary to {self.summary_cache_path}")
            except Exception as e:
                print(f"[Prompts v2] Warning: could not cache summary: {e}")

        return summary

    def generate_atoms_for_role(
        self,
        dataset_name: str,
        role: str,
        count: int = 5,
        examples: Optional[str] = None,
        summary: Optional[str] = None,
        existing_atoms: Optional[List[str]] = None,
    ) -> Dict[int, str]:
        """
        Generate `count` descriptive atoms for `role`. Reuses pre-fetched
        examples/summary when provided (avoids redundant LLM calls).
        Mutates `existing_atoms` in place so siblings see what's already been generated.
        """
        if examples is None:
            examples = self._get_dataset_examples(dataset_name, n=5)
        if summary is None:
            summary = self._generate_dataset_summary(dataset_name, examples)
        if existing_atoms is None:
            existing_atoms = []

        role_goals = {
            "reasoner": "guide the step-by-step thinking process",
            "verifier": "critique reasoning and surface errors before finalization",
            "answerer": "format and present the final output concisely and correctly",
            "router": "analyze the question and select the most appropriate reasoning approach or agent",
            "orchestrator": "decompose the problem into sub-tasks for parallel or sequential resolution",
            "aggregator": "resolve conflicts between multiple candidate answers or sub-task results",
        }
        goal = role_goals.get(role, "solve the task")

        n = min(count, len(self.strategies))
        selected_strategies = random.sample(list(self.strategies.items()), n)

        print(f"[Prompts v2] Generating {n} {role} atoms for {dataset_name}...")

        atoms: Dict[int, str] = {}
        for i, (strat_name, strat_desc) in enumerate(selected_strategies):
            existing_block = (
                "\n".join(f"- {a}" for a in existing_atoms) if existing_atoms else "(none yet)"
            )
            meta_prompt = (
                f"You are an expert prompt engineer producing a reusable system instruction.\n\n"
                f"## Task Analysis\n{summary}\n\n"
                f"## Real Examples (each shows the full task end-to-end: problem, reasoning, answer)\n"
                f"{examples}\n"
                f"NOTE: These examples cover every stage of the task. For this generation, "
                f"focus only on the parts relevant to a '{role}' agent and ignore the rest.\n\n"
                f"## Role\nWrite a system instruction that helps a '{role}' agent {goal}.\n\n"
                f"## Strategy\n{strat_desc}\n\n"
                f"## Already-generated atoms (avoid producing semantically similar instructions)\n"
                f"{existing_block}\n\n"
                f"## Format requirements\n"
                f"- 2-3 complete sentences, roughly 40-80 words. Be descriptive and specific within that length, but do NOT pad. End every sentence with a period; do not stop mid-sentence.\n"
                f"- Name a concrete technique, state when to apply it, and describe the specific action to take.\n"
                f"- The instruction will be appended verbatim to the agent's system prompt for every question, "
                f"so phrase it as a standing rule the agent should follow (e.g., \"When the question involves...,\" \"Before finalizing your answer, ...\").\n"
                f"- The instruction should be tailored to the *substance* of the task shown above (the kinds of "
                f"inputs, the discriminating features, the typical pitfalls). However, do NOT name the dataset, "
                f"benchmark, or exam (do not write \"{dataset_name}\", \"vignette\", \"USMLE\", \"GSM8K\", \"HotpotQA\", "
                f"\"this benchmark\", \"this dataset\", etc.). Refer to the inputs by what they are (\"clinical case\", "
                f"\"word problem\", \"multi-hop question\"), not by their dataset label.\n"
                f"- Output ONLY the instruction text. No quotes, no labels (e.g. 'Instruction:'), no preamble, no trailing commentary."
            )

            max_retries = 3
            atom_text = None
            # 512 tokens leaves comfortable headroom for 40-80 word atoms plus
            # any internal preamble/reasoning Claude emits before the instruction.
            atom_token_budget = 512
            for attempt in range(max_retries):
                try:
                    response = self.worker.answer_direct(
                        "Generate instruction", prompt_suffix=meta_prompt,
                        tokens=atom_token_budget,
                    )
                    if response:
                        cleaned = self._clean_atom_text(response)
                        leak_reason = self._dataset_name_leak(cleaned, dataset_name)
                        truncated = self._looks_truncated(cleaned)
                        if cleaned and len(cleaned) >= 40 and not leak_reason and not truncated:
                            atom_text = cleaned
                            break
                        # Otherwise fall through to retry with a logged reason
                        if attempt < max_retries - 1:
                            reason = (
                                "empty/short" if not cleaned or len(cleaned) < 40
                                else (f"dataset/benchmark name leaked ({leak_reason})" if leak_reason
                                      else "looks truncated")
                            )
                            print(f"  Rejecting {role} atom {i+1} ({reason}), retrying ({attempt+1}/{max_retries})...")
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  Error generating {role} atom {i+1}: {e}, retrying ({attempt+1}/{max_retries})...")
                    else:
                        print(f"  Failed to generate {role} atom {i+1} after {max_retries} attempts: {e}")

            key = 100 + i
            if atom_text:
                atoms[key] = atom_text
                existing_atoms.append(atom_text)
            else:
                print(f"  Skipping {role} atom {i+1} (empty/too short after {max_retries} attempts)")

        return atoms

    def generate_all_atoms(self, dataset_name: str) -> Dict[str, Dict[int, str]]:
        """
        Generate the full atom set for a dataset.

        Pool sizes (v2):
          reasoner = 10  (5 reasoner + 3 router + 2 orchestrator)
          verifier = 7
          answerer = 7   (4 answerer + 3 aggregator)

        Examples and the dataset summary are computed once and shared across all roles
        so subsequent role generations don't pay the LLM cost of reanalyzing the dataset.
        """
        examples = self._get_dataset_examples(dataset_name, n=5)
        summary = self._generate_dataset_summary(dataset_name, examples)

        # Cross-role dedup: every generated atom is appended here so later atoms
        # (in any role) see it in the "already-generated" block.
        all_existing: List[str] = []

        # --- Reasoner pool: 10 atoms ---
        reasoner: Dict[int, str] = {}
        idx = 100
        for concept, n in [("reasoner", 5), ("router", 3), ("orchestrator", 2)]:
            generated = self.generate_atoms_for_role(
                dataset_name, concept, n, examples, summary, all_existing
            )
            for text in generated.values():
                if text and text.strip():
                    reasoner[idx] = text
                    idx += 1

        # --- Verifier pool: 7 atoms ---
        verifier: Dict[int, str] = {}
        idx = 100
        generated = self.generate_atoms_for_role(
            dataset_name, "verifier", 7, examples, summary, all_existing
        )
        for text in generated.values():
            if text and text.strip():
                verifier[idx] = text
                idx += 1

        # --- Answerer pool: 7 atoms ---
        answerer: Dict[int, str] = {}
        idx = 100
        for concept, n in [("answerer", 4), ("aggregator", 3)]:
            generated = self.generate_atoms_for_role(
                dataset_name, concept, n, examples, summary, all_existing
            )
            for text in generated.values():
                if text and text.strip():
                    answerer[idx] = text
                    idx += 1

        # Final cleaning pass (defense-in-depth; atoms are already cleaned individually)
        def clean_atoms_dict(d: Dict[int, str]) -> Dict[int, str]:
            return {k: (self._clean_atom_text(v) if v else v) for k, v in d.items()}

        return {
            "reasoner": clean_atoms_dict(reasoner),
            "verifier": clean_atoms_dict(verifier),
            "answerer": clean_atoms_dict(answerer),
        }
