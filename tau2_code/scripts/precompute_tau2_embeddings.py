"""Precompute MetaCLIP-H14 embeddings for tau2 task descriptions.

Run ONCE before training/eval. Loads the embedder model on GPU exactly one time,
embeds every task description across the chosen tau2 domains, and writes
embeddings_cache/tau2_<domain>_embeddings.npz keyed by the same md5(text)[:16]
hash that MetaCLIPEmbedder._compute_hash uses at runtime.

After this runs successfully, every subsequent OpenRouterWorker / LLMWorker
construction will read embeddings from disk and never touch CUDA, fixing the
"CUDA out of memory" you hit when running --workers 10 in parallel.

Usage:
    python tau2_code/scripts/precompute_tau2_embeddings.py --all
    python tau2_code/scripts/precompute_tau2_embeddings.py --domains airline retail
    python tau2_code/scripts/precompute_tau2_embeddings.py --all --force      # recompute
"""
import argparse
import hashlib
import os
import sys
from typing import List

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Quiet tau2's loguru spam so the progress bar is readable.
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
except Exception:
    pass

from tqdm import tqdm
from tau2_code.src.dataset import _render_task_description, Tau2Dataset


_DEFAULT_DOMAINS = ["mock", "airline", "retail", "telecom"]
_CACHE_DIR = os.path.join(_REPO, "embeddings_cache")


def _hash_text(text: str) -> str:
    """Must match MetaCLIPEmbedder._compute_hash exactly."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _collect_tasks(domain: str):
    """Return list of (hash, description_text) over BOTH train and test splits.

    We embed the union so a single cache file covers training, eval, and
    transfer experiments without recomputation.
    """
    seen_hashes = set()
    out = []
    for split in ("train", "test"):
        try:
            ds = Tau2Dataset(domain=domain, split=split)
        except Exception as e:
            # mock has no split file - falls back to base internally; ignore the second iter.
            if split == "test":
                continue
            raise
        for task in ds.tasks:
            text = _render_task_description(task)
            h = _hash_text(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            out.append((h, text))
    return out


def _write_cache(domain: str, hashes: List[str], embeddings: np.ndarray, force: bool):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    out_path = os.path.join(_CACHE_DIR, f"tau2_{domain}_embeddings.npz")
    if os.path.exists(out_path) and not force:
        print(f"[precompute] {out_path} already exists. Use --force to overwrite.")
        return out_path
    np.savez(out_path, hashes=np.array(hashes), embeddings=embeddings.astype(np.float32))
    print(f"[precompute] wrote {len(hashes)} embeddings -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Precompute tau2 task description embeddings.")
    parser.add_argument("--domains", nargs="*", default=None,
                        choices=_DEFAULT_DOMAINS,
                        help="Domains to precompute. Default: all.")
    parser.add_argument("--all", action="store_true", help="Process all tau2 domains.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing cache files.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Embed batch size on the GPU.")
    args = parser.parse_args()

    domains = _DEFAULT_DOMAINS if (args.all or not args.domains) else args.domains

    # Load the embedder ONCE. Passing use_precomputed=False forces a real model
    # load - we want that here because we're populating the cache.
    print("[precompute] loading MetaCLIP-H14 on GPU (one-time)...")
    from agents_system.worker import MetaCLIPEmbedder
    embedder = MetaCLIPEmbedder(target_dim=None, use_precomputed=False)
    embedder._init_embedder()
    print(f"[precompute] embedder ready; dim={embedder.embedding_dim}")

    for domain in domains:
        print(f"\n[precompute] === domain: {domain} ===")
        items = _collect_tasks(domain)
        print(f"[precompute] {len(items)} unique task descriptions")
        if not items:
            continue

        out_path = os.path.join(_CACHE_DIR, f"tau2_{domain}_embeddings.npz")
        if os.path.exists(out_path) and not args.force:
            print(f"[precompute] {out_path} already exists, skipping (use --force to overwrite).")
            continue

        hashes = []
        vectors = []
        for h, text in tqdm(items, desc=f"embedding {domain}", unit="task"):
            vec = embedder.embed(text)
            hashes.append(h)
            vectors.append(vec)
        vectors = np.stack(vectors, axis=0)
        _write_cache(domain, hashes, vectors, force=args.force)

    print(f"\n[precompute] done. Cache dir: {_CACHE_DIR}")


if __name__ == "__main__":
    main()
