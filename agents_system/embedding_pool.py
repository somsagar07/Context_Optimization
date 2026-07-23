"""Thread-safe pool of MetaCLIP model copies for parallel embedding.

Standard MetaCLIPEmbedder shares ONE model across all threads. That serializes
GPU inference when many worker threads need embeddings simultaneously (common
in per-turn rollouts where every turn is a cache miss).

EmbeddingPool loads K independent model copies, each with its own CUDA stream.
Worker threads checkout a model, compute the embedding, and return it. This
gives up to K-way GPU parallelism.

Usage:
    pool = EmbeddingPool(pool_size=4)
    # Drop into any MetaCLIPEmbedder via class method:
    MetaCLIPEmbedder.set_pool(pool)
    # Now all embedder.embed() calls go through the pool.
"""
from __future__ import annotations

import hashlib
import os
import queue
import threading
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


_MODEL_NAME = "facebook/metaclip-h14-fullcc2.5b"


class _PooledModel:
    """One model copy with its own CUDA stream."""
    __slots__ = ("model", "processor", "device", "stream", "max_length")

    def __init__(self, model, processor, device, stream, max_length):
        self.model = model
        self.processor = processor
        self.device = device
        self.stream = stream
        self.max_length = max_length


class EmbeddingPool:
    """Thread-safe pool of K MetaCLIP model copies."""

    def __init__(
        self,
        pool_size: int = 2,
        device: Optional[str] = None,
        target_dim: Optional[int] = None,
    ):
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")

        self.pool_size = pool_size
        self.target_dim = target_dim
        self.embedding_dim: int = 1024
        self.projection: Optional[np.ndarray] = None
        self._pool: queue.Queue[_PooledModel] = queue.Queue()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._base_device = device

        _quiet = bool(os.environ.get("CO_QUIET"))
        if not _quiet:
            print(f"[EmbeddingPool] Loading {pool_size} MetaCLIP copies on {device}...")

        processor = AutoProcessor.from_pretrained(_MODEL_NAME)
        max_length = 77
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "model_max_length"):
            max_length = processor.tokenizer.model_max_length

        for i in range(pool_size):
            model = AutoModel.from_pretrained(_MODEL_NAME)
            model.to(device).eval()

            stream = torch.cuda.Stream(device=device) if device.startswith("cuda") else None
            pm = _PooledModel(model, processor, device, stream, max_length)
            self._pool.put(pm)

        with torch.no_grad():
            inputs = processor(text=["test"], return_tensors="pt").to(device)
            first_pm = self._pool.get()
            try:
                out = self._get_text_features(first_pm.model, inputs)
                self.embedding_dim = out.shape[1]
            finally:
                self._pool.put(first_pm)

        if target_dim and self.embedding_dim != target_dim:
            self._init_projection(self.embedding_dim)

        if not _quiet:
            print(f"[EmbeddingPool] Ready. dim={self.embedding_dim}, pool={pool_size}")

        # Share the precomputed cache from MetaCLIPEmbedder (if loaded).
        from .worker import MetaCLIPEmbedder
        if not MetaCLIPEmbedder._cache_loaded:
            MetaCLIPEmbedder._load_precomputed_cache()

    @staticmethod
    def _to_text_feature_tensor(outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs
        for attr in ("text_embeds", "pooler_output"):
            t = getattr(outputs, attr, None)
            if isinstance(t, torch.Tensor):
                return t
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if isinstance(last_hidden, torch.Tensor):
            return last_hidden[:, 0, :]
        raise RuntimeError(f"Unexpected output type: {type(outputs)}")

    @classmethod
    def _get_text_features(cls, model, inputs):
        if hasattr(model, "get_text_features"):
            out = model.get_text_features(**inputs)
        else:
            out = model(**inputs)
        return cls._to_text_feature_tensor(out)

    def _init_projection(self, input_dim: int):
        np.random.seed(42)
        self.projection = np.random.randn(self.target_dim, input_dim).astype(np.float32)
        self.projection = self.projection / np.linalg.norm(self.projection, axis=0, keepdims=True)

    def embed(self, text: str) -> np.ndarray:
        """Compute embedding, using precomputed cache when available."""
        from .worker import MetaCLIPEmbedder

        # Cache lookup first (O(1), no model needed)
        if MetaCLIPEmbedder._precomputed_cache:
            text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
            cached = MetaCLIPEmbedder._precomputed_cache.get(text_hash)
            if cached is not None:
                MetaCLIPEmbedder._cache_stats["hits"] += 1
                embedding = cached.copy()
                if self.target_dim and embedding.shape[0] != self.target_dim:
                    if self.projection is None:
                        self._init_projection(embedding.shape[0])
                    embedding = self.projection @ embedding
                return embedding.astype(np.float32)
            MetaCLIPEmbedder._cache_stats["misses"] += 1

        # Checkout a model copy (blocks if all K are busy)
        pm = self._pool.get()
        try:
            ctx = torch.cuda.stream(pm.stream) if pm.stream else _nullcontext()
            with ctx:
                with torch.no_grad():
                    inputs = pm.processor(
                        text=[text],
                        return_tensors="pt",
                        truncation=True,
                        padding=False,
                        max_length=pm.max_length,
                    ).to(pm.device)
                    outputs = self._get_text_features(pm.model, inputs)
                    if pm.stream:
                        pm.stream.synchronize()
                    embedding = outputs.cpu().numpy().flatten()
        finally:
            self._pool.put(pm)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if self.target_dim and embedding.shape[0] != self.target_dim:
            if self.projection is None:
                self._init_projection(embedding.shape[0])
            embedding = self.projection @ embedding

        return embedding.astype(np.float32)

    def get_dimension(self) -> int:
        return self.target_dim if self.target_dim else self.embedding_dim

    def ensure_loaded(self):
        pass  # Models are loaded eagerly at init.


class _nullcontext:
    """Minimal context manager for non-CUDA fallback (Python 3.6 compat)."""
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
