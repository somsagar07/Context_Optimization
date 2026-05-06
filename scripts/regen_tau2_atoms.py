"""Regenerate tau2 atoms via AtomGeneratorV2 and run a leak audit immediately.

Used as a one-shot tool after fixing the generator placeholder. Call:
    python scripts/regen_tau2_atoms.py tau2_telecom tau2_mock
"""
import json
import os
import re
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO, ".env"))

from agents_system.worker import OpenRouterWorker
from prompts import library

_LEAK_PATTERNS = [
    (r"\bverbatim\b", "verbatim"),
    (r"\bcanonical\b", "canonical"),
    (r"resolution directive", "resolution directive"),
    (r"target output", "target output"),
    (r"word.for.word", "word-for-word"),
    (r"one plain.text line", "plain-text line"),
    (r"do not paraphrase", "do not paraphrase"),
    (r"copy(?:ed)? verbatim", "copy verbatim"),
    (r"strip any troubleshooting", "strip troubleshooting"),
    (r"please provide more details?", "stall: please provide more details"),
    (r"expected actions?:", "label: expected actions"),
    (r"must communicate:", "label: must communicate"),
    (r"nl assertions?:", "label: nl assertions"),
]


def audit(dataset_name):
    p = os.path.join(_REPO, "prompts", "generated_v2", dataset_name, "atoms.json")
    if not os.path.exists(p):
        print(f"  [audit] missing: {p}")
        return None
    d = json.load(open(p))
    total = sum(len(v) for v in d.values())
    flagged = 0
    hits_by_role = {}
    for role, atoms in d.items():
        for k, v in atoms.items():
            t = (v or "").lower()
            hits = [name for pat, name in _LEAK_PATTERNS if re.search(pat, t)]
            if hits:
                flagged += 1
                hits_by_role.setdefault(role, []).append((k, hits, v[:140]))
    print(f"  [audit] {dataset_name}: flagged {flagged}/{total}")
    for role, items in hits_by_role.items():
        for k, hits, snip in items:
            print(f"    [{role}#{k}] {hits}")
            print(f"      >> {snip!r}")
    return flagged


def regen(dataset_name, model_id):
    print(f"=== regenerating {dataset_name} via {model_id} ===")
    worker = OpenRouterWorker(model_name=model_id)
    library.load_or_create_atoms(dataset_name, worker=worker)
    return audit(dataset_name)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["tau2_telecom", "tau2_mock"]
    model_id = os.environ.get("REGEN_MODEL", "anthropic/claude-opus-4.7")
    for ds in targets:
        n = regen(ds, model_id)
        if n is None:
            print(f"  -> {ds}: SKIPPED (no atoms file)")
        elif n == 0:
            print(f"  -> {ds}: CLEAN")
        else:
            print(f"  -> {ds}: STILL POISONED ({n} flagged)")
