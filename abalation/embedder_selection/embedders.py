"""
Unified embedding interface for fair ablation studies.
All embedders implement the same interface for HuggingFace and API workers.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class BaseEmbedder(ABC):
    """Base class for all embedders."""
    
    def __init__(self, name: str, target_dim: Optional[int] = None):
        self.name = name
        self.embedding_dim = None  # Set by subclass
        self.target_dim = target_dim  # Projection target if specified
        self.projection = None
        self._initialized = False
    
    @abstractmethod
    def _embed(self, text: str) -> np.ndarray:
        """Internal embedding method (returns raw embedding)."""
        pass
    
    @abstractmethod
    def _init_embedder(self):
        """Initialize the embedder model (called once)."""
        pass
    
    def embed(self, text: str) -> np.ndarray:
        """
        Public interface: returns embedding in target_dim.
        If target_dim is set, projects to that dimension.
        """
        if not self._initialized:
            self._init_embedder()
            self._initialized = True
        
        embedding = self._embed(text)
        
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Project if needed
        if self.target_dim and embedding.shape[0] != self.target_dim:
            if self.projection is None:
                self._init_projection(embedding.shape[0])
            embedding = self.projection @ embedding
        
        return embedding.astype(np.float32)
    
    def _init_projection(self, input_dim: int):
        """Initialize projection matrix (fixed random projection)."""
        np.random.seed(42)  # Deterministic
        self.projection = np.random.randn(self.target_dim, input_dim).astype(np.float32)
        # Normalize projection matrix
        self.projection = self.projection / np.linalg.norm(self.projection, axis=0, keepdims=True)
    
    def get_dimension(self) -> int:
        """Get the output embedding dimension."""
        if not self._initialized:
            self._init_embedder()
            self._initialized = True
        return self.target_dim if self.target_dim else self.embedding_dim


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence-transformers based embedder."""
    
    def __init__(self, model_name: str, target_dim: Optional[int] = None):
        super().__init__(f"sentence-{model_name.split('/')[-1]}", target_dim)
        self.model_name = model_name
        self.model = None
    
    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading sentence-transformer: {self.model_name}...", end=" ", flush=True)
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            if self.target_dim and self.embedding_dim != self.target_dim:
                self._init_projection(self.embedding_dim)
            output_dim = self.target_dim if self.target_dim else self.embedding_dim
            print(f"✓ Loaded. Dimension: {self.embedding_dim} -> {output_dim}")
        except ImportError:
            raise ImportError("sentence-transformers not installed: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {e}")
    
    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)


class CLIPEmbedder(BaseEmbedder):
    """CLIP-based embedder (uses text encoder)."""
    
    def __init__(self, model_name: str, target_dim: Optional[int] = None):
        super().__init__(f"clip-{model_name.split('/')[-1]}", target_dim)
        self.model_name = model_name
        self.model = None
        self.device = None
    
    def _init_embedder(self):
        try:
            import clip
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # CLIP expects full model name like "ViT-B/32", not just "32"
            # self.model_name should already be like "ViT-B/32" from the factory
            print(f"  Loading CLIP: {self.model_name}...", end=" ", flush=True)
            self.model, _ = clip.load(self.model_name, device=self.device)
            self.model.eval()
            # Get embedding dim from model
            with torch.no_grad():
                dummy = clip.tokenize(["test"]).to(self.device)
                dummy_embed = self.model.encode_text(dummy)
                self.embedding_dim = dummy_embed.shape[1]
            if self.target_dim and self.embedding_dim != self.target_dim:
                self._init_projection(self.embedding_dim)
            output_dim = self.target_dim if self.target_dim else self.embedding_dim
            print(f"✓ Loaded. Dimension: {self.embedding_dim} -> {output_dim}")
        except ImportError:
            raise ImportError("clip not installed: pip install git+https://github.com/openai/CLIP.git")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP {self.model_name}: {e}")
    
    def _embed(self, text: str) -> np.ndarray:
        import clip
        with torch.no_grad():
            tokens = clip.tokenize([text], truncate=True).to(self.device)
            embedding = self.model.encode_text(tokens)
            return embedding.cpu().numpy().flatten()


class SigLIPEmbedder(BaseEmbedder):
    """SigLIP embedder."""
    
    def __init__(self, model_name: str, target_dim: Optional[int] = None):
        super().__init__(f"siglip-{model_name.split('/')[-1]}", target_dim)
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = None
    
    def _init_embedder(self):
        try:
            try:
                from transformers import AutoProcessor, AutoModel
            except ImportError as ie:
                raise ImportError(f"transformers not installed: pip install transformers. Original: {ie}")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Loading SigLIP: {self.model_name}...", end=" ", flush=True)
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
            except Exception as model_error:
                raise RuntimeError(f"Failed to load model {self.model_name} from HuggingFace: {model_error}. "
                                 f"Make sure the model exists and you have access (run: huggingface-cli login)")
            self.model.to(self.device).eval()
            
            # Get embedding dimension
            with torch.no_grad():
                inputs = self.processor(text=["test"], return_tensors="pt").to(self.device)
                outputs = self.model.get_text_features(**inputs)
                self.embedding_dim = outputs.shape[1]
            
            # Get max sequence length from tokenizer (SigLIP models use 64)
            # The processor.tokenizer.model_max_length is the actual limit enforced by the model
            self.max_length = self.processor.tokenizer.model_max_length
            
            if self.target_dim and self.embedding_dim != self.target_dim:
                self._init_projection(self.embedding_dim)
            output_dim = self.target_dim if self.target_dim else self.embedding_dim
            print(f"✓ Loaded. Dimension: {self.embedding_dim} -> {output_dim}, Max length: {self.max_length}")
        except ImportError as ie:
            raise ImportError(f"transformers not installed: pip install transformers. Original error: {ie}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SigLIP {self.model_name}: {e}")
    
    def _embed(self, text: str) -> np.ndarray:
        """Embed text with proper truncation to model's max length."""
        max_len = self.max_length  # Set during initialization from tokenizer.model_max_length
        
        with torch.no_grad():
            inputs = self.processor(
                text=[text], 
                return_tensors="pt", 
                truncation=True,
                padding=False,
                max_length=max_len
            ).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            return outputs.cpu().numpy().flatten()


class JinaCLIPEmbedder(BaseEmbedder):
    """Jina CLIP v2 embedder - multilingual multimodal embeddings."""
    
    def __init__(self, model_name: str = "jinaai/jina-clip-v2", target_dim: Optional[int] = None):
        super().__init__("jina-clip-v2", target_dim)
        self.model_name = model_name
        self.model = None
        self.device = None
    
    def _init_embedder(self):
        try:
            try:
                from transformers import AutoModel
            except ImportError as ie:
                raise ImportError(f"transformers not installed: pip install transformers. Original: {ie}")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Loading Jina CLIP v2: {self.model_name}...", end=" ", flush=True)
            try:
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception as model_error:
                raise RuntimeError(f"Failed to load model {self.model_name} from HuggingFace: {model_error}. "
                                 f"Make sure the model exists and you have access (run: huggingface-cli login)")
            self.model.to(self.device).eval()
            
            # Get embedding dimension (Jina CLIP v2 has 1024 dim, can be truncated to 64)
            with torch.no_grad():
                test_embedding = self.model.encode_text("test", normalize_embeddings=False, convert_to_numpy=True)
                # encode_text returns 1D array for single string, or 2D for list
                if len(test_embedding.shape) == 1:
                    self.embedding_dim = test_embedding.shape[0]
                else:
                    self.embedding_dim = test_embedding.shape[1]
            
            if self.target_dim and self.embedding_dim != self.target_dim:
                self._init_projection(self.embedding_dim)
            output_dim = self.target_dim if self.target_dim else self.embedding_dim
            print(f"✓ Loaded. Dimension: {self.embedding_dim} -> {output_dim}")
        except ImportError as ie:
            raise ImportError(f"transformers not installed: pip install transformers. Original error: {ie}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Jina CLIP v2 {self.model_name}: {e}")
    
    def _embed(self, text: str) -> np.ndarray:
        """Embed text using Jina CLIP v2's encode_text method."""
        with torch.no_grad():
            # Jina CLIP v2 encode_text accepts string or list of strings
            # Supports up to 8192 tokens for text
            # normalize_embeddings=False because base class handles normalization
            # convert_to_numpy=True to get numpy array directly
            embedding = self.model.encode_text(
                text, 
                normalize_embeddings=False,
                convert_to_numpy=True
            )
            # encode_text returns 1D array for single string
            return embedding


class FLAVAEmbedder(BaseEmbedder):
    """FLAVA (Facebook Language-Vision-Audio) embedder."""
    
    def __init__(self, model_name: str = "facebook/flava-full", target_dim: Optional[int] = None):
        super().__init__("flava-full", target_dim)
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = None
    
    def _init_embedder(self):
        try:
            try:
                from transformers import AutoProcessor, AutoModel
            except ImportError as ie:
                raise ImportError(f"transformers not installed: pip install transformers. Original: {ie}")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Loading FLAVA: {self.model_name}...", end=" ", flush=True)
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
            except Exception as model_error:
                raise RuntimeError(f"Failed to load model {self.model_name} from HuggingFace: {model_error}. "
                                 f"Make sure the model exists and you have access (run: huggingface-cli login)")
            self.model.to(self.device).eval()
            
            # Get embedding dimension from text features
            # FLAVA returns [batch_size, sequence_length, hidden_dim], need to pool
            with torch.no_grad():
                inputs = self.processor(text=["test"], return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model.get_text_features(**inputs)
                # FLAVA returns per-token embeddings, use the hidden_dim (last dimension)
                if len(outputs.shape) == 3:
                    # [batch, seq_len, hidden_dim] - use hidden_dim
                    self.embedding_dim = outputs.shape[2]
                else:
                    # [batch, hidden_dim] - already pooled
                    self.embedding_dim = outputs.shape[1]
            
            # Get max sequence length from tokenizer
            if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'model_max_length'):
                self.max_length = self.processor.tokenizer.model_max_length
            else:
                self.max_length = 512  # Default for FLAVA
            
            if self.target_dim and self.embedding_dim != self.target_dim:
                self._init_projection(self.embedding_dim)
            output_dim = self.target_dim if self.target_dim else self.embedding_dim
            print(f"✓ Loaded. Dimension: {self.embedding_dim} -> {output_dim}, Max length: {self.max_length}")
        except ImportError as ie:
            raise ImportError(f"transformers not installed: pip install transformers. Original error: {ie}")
        except Exception as e:
            raise RuntimeError(f"Failed to load FLAVA {self.model_name}: {e}")
    
    def _embed(self, text: str) -> np.ndarray:
        """Embed text using FLAVA's text encoder."""
        max_len = self.max_length
        
        with torch.no_grad():
            inputs = self.processor(
                text=[text], 
                return_tensors="pt", 
                truncation=True,
                padding=False,
                max_length=max_len
            ).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            
            # FLAVA returns [batch_size, sequence_length, hidden_dim]
            # Need to pool over sequence dimension (mean pooling)
            if len(outputs.shape) == 3:
                # [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
                outputs = outputs.mean(dim=1)
            
            return outputs.cpu().numpy().flatten()


class ModelHiddenStateEmbedder(BaseEmbedder):
    """Uses model's hidden states (original HF approach)."""
    
    def __init__(self, model, tokenizer, target_dim: Optional[int] = None):
        super().__init__("model-hidden-states", target_dim)
        self.model = model
        self.tokenizer = tokenizer
        self.device = None
    
    def _init_embedder(self):
        self.device = next(self.model.parameters()).device
        # Get embedding dim
        with torch.no_grad():
            dummy = self.tokenizer(
                "test", 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            outputs = self.model(**dummy, output_hidden_states=True)
            self.embedding_dim = outputs.hidden_states[-1].shape[-1]
        if self.target_dim and self.embedding_dim != self.target_dim:
            self._init_projection(self.embedding_dim)
        output_dim = self.target_dim if self.target_dim else self.embedding_dim
        print(f"  Model hidden states. Dimension: {self.embedding_dim} -> {output_dim}")
    
    def _embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        
        return embedding


class E5Embedder(BaseEmbedder):
    """E5 embedding model (sentence-transformer variant)."""
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2", target_dim: Optional[int] = None):
        super().__init__(f"e5-{model_name.split('/')[-1]}", target_dim)
        self.model_name = model_name
        self.model = None
    
    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading E5: {self.model_name}...", end=" ", flush=True)
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            if self.target_dim and self.embedding_dim != self.target_dim:
                self._init_projection(self.embedding_dim)
            output_dim = self.target_dim if self.target_dim else self.embedding_dim
            print(f"✓ Loaded. Dimension: {self.embedding_dim} -> {output_dim}")
        except ImportError:
            raise ImportError("sentence-transformers not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to load E5 {self.model_name}: {e}")
    
    def _embed(self, text: str) -> np.ndarray:
        # E5 models need prefix
        if "e5" in self.model_name.lower():
            text = f"query: {text}"
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)


# Factory function
def create_embedder(embedder_type: str, **kwargs) -> BaseEmbedder:
    """Factory to create embedders."""
    target_dim = kwargs.get("target_dim", 512)  # Default target dimension
    
    if embedder_type.startswith("sentence-"):
        model_name = embedder_type.replace("sentence-", "")
        return SentenceTransformerEmbedder(model_name, target_dim=target_dim)
    
    elif embedder_type.startswith("clip-"):
        # Remove "clip-" prefix (5 characters)
        # Result should be like "ViT-B/32" which CLIP library expects
        model_name = embedder_type[5:]
        return CLIPEmbedder(model_name, target_dim=target_dim)
    
    elif embedder_type.startswith("siglip-"):
        # Remove "siglip-" prefix (7 characters)
        model_name = embedder_type[7:]
        return SigLIPEmbedder(model_name, target_dim=target_dim)
    
    elif embedder_type.startswith("e5-"):
        model_name = embedder_type.replace("e5-", "") or "intfloat/e5-base-v2"
        return E5Embedder(model_name, target_dim=target_dim)
    
    elif embedder_type.startswith("jina-clip"):
        # Handle jina-clip-v2 or jina-clip-<model_name>
        if embedder_type == "jina-clip-v2":
            model_name = "jinaai/jina-clip-v2"
        else:
            model_name = embedder_type.replace("jina-clip-", "")
        return JinaCLIPEmbedder(model_name, target_dim=target_dim)
    
    elif embedder_type.startswith("flava"):
        # Handle flava-full or flava-<model_name>
        if embedder_type == "flava-full" or embedder_type == "flava":
            model_name = "facebook/flava-full"
        else:
            model_name = embedder_type.replace("flava-", "")
        return FLAVAEmbedder(model_name, target_dim=target_dim)
    
    elif embedder_type == "model-hidden":
        return ModelHiddenStateEmbedder(
            kwargs["model"], 
            kwargs["tokenizer"], 
            target_dim=target_dim
        )
    
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


# Predefined embedder configurations
EMBEDDER_CONFIGS = {
    # Sentence-transformers (fast, general purpose)
    "all-MiniLM-L6-v2": {
        "type": "sentence-all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Fast, lightweight sentence-transformer"
    },
    "all-MiniLM-L12-v2": {
        "type": "sentence-all-MiniLM-L12-v2",
        "dim": 384,
        "description": "Larger MiniLM variant"
    },
    "all-mpnet-base-v2": {
        "type": "sentence-all-mpnet-base-v2",
        "dim": 768,
        "description": "Higher quality sentence-transformer"
    },
    "sentence-t5-base": {
        "type": "sentence-sentence-t5-base",
        "dim": 768,
        "description": "T5-based sentence-transformer"
    },
    
    # E5 models (modern, high quality)
    "e5-base": {
        "type": "e5-intfloat/e5-base-v2",
        "dim": 768,
        "description": "E5 base model (modern embedding)"
    },
    
    # Vision-language models - CLIP variants
    "clip-base": {
        "type": "clip-ViT-B/32",
        "dim": 512,
        "description": "CLIP base ViT-B/32"
    },
    "clip-large": {
        "type": "clip-ViT-L/14",
        "dim": 768,
        "description": "CLIP large ViT-L/14"
    },
    "clip-base-patch16": {
        "type": "clip-ViT-B/16",
        "dim": 512,
        "description": "CLIP base ViT-B/16"
    },
    
    # SigLIP variants
    "siglip-base": {
        "type": "siglip-google/siglip-base-patch16-224",
        "dim": 768,
        "description": "SigLIP base patch16-224"
    },
    "siglip-large": {
        "type": "siglip-google/siglip-large-patch16-384",
        "dim": 1024,
        "description": "SigLIP large patch16-384"
    },
    # Note: patch32 and so400m variants may not exist - only base and large are verified
    
    # Jina CLIP v2 (multilingual multimodal)
    "jina-clip-v2": {
        "type": "jina-clip-v2",
        "dim": 1024,
        "description": "Jina CLIP v2 - multilingual multimodal (supports 89 languages, 8192 tokens)"
    },
    
    # FLAVA (Facebook multimodal)
    "flava-full": {
        "type": "flava-full",
        "dim": 768,
        "description": "FLAVA full model - unified vision-language model"
    },
}

