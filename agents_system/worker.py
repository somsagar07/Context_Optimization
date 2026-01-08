import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel
import numpy as np
import sys
import os
import requests
sys.path.append('..')
from configs.base import LLM_MODEL_NAME, DEVICE

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # Try loading from current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass


class MetaCLIPEmbedder:
    """MetaCLIP-H14 embedder for consistent embeddings across API and HuggingFace modes."""
    
    def __init__(self, target_dim: int = None):
        """
        Initialize MetaCLIP-H14 embedder.
        
        Args:
            target_dim: Target embedding dimension (None = use native 1024D, no projection)
        """
        self.model_name = "facebook/metaclip-h14-fullcc2.5b"
        self.target_dim = target_dim
        self.processor = None
        self.model = None
        self.device = None
        self.embedding_dim = None
        self.projection = None
        self._initialized = False
    
    def _init_embedder(self):
        """Initialize the MetaCLIP model."""
        if self._initialized:
            return
        
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading MetaCLIP-H14 embedder: {self.model_name}...", end=" ", flush=True)
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device).eval()
            
            # Get embedding dimension
            with torch.no_grad():
                inputs = self.processor(text=["test"], return_tensors="pt").to(self.device)
                outputs = self.model.get_text_features(**inputs)
                self.embedding_dim = outputs.shape[1]
            
            # Get max sequence length from tokenizer
            if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'model_max_length'):
                self.max_length = self.processor.tokenizer.model_max_length
            else:
                self.max_length = 77  # Default for CLIP-like models
            
            # Initialize projection if needed
            if self.target_dim and self.embedding_dim != self.target_dim:
                self._init_projection(self.embedding_dim)
            
            output_dim = self.target_dim if self.target_dim else self.embedding_dim
            print(f"✓ Loaded. Dimension: {self.embedding_dim} -> {output_dim}")
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MetaCLIP-H14 embedder: {e}. "
                             f"Make sure the model exists and you have access (run: huggingface-cli login)")
    
    def _init_projection(self, input_dim: int):
        """Initialize projection matrix (fixed random projection)."""
        np.random.seed(42)  # Deterministic
        self.projection = np.random.randn(self.target_dim, input_dim).astype(np.float32)
        # Normalize projection matrix
        self.projection = self.projection / np.linalg.norm(self.projection, axis=0, keepdims=True)
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text using MetaCLIP-H14.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array embedding (1024D if target_dim=None, otherwise target_dim)
        """
        if not self._initialized:
            self._init_embedder()
        
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
            embedding = outputs.cpu().numpy().flatten()
        
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
    
    def get_dimension(self) -> int:
        """Get the output embedding dimension."""
        if not self._initialized:
            self._init_embedder()
        return self.target_dim if self.target_dim else self.embedding_dim


class LLMWorker:
    """LLM-based worker that handles reasoning, verification, and answering."""
    
    def __init__(self, model_name: str = None):
        # Use provided model_name or fall back to config default
        self.model_name = model_name or LLM_MODEL_NAME
        self.device = DEVICE
        
        print(f"Loading Worker Model: {self.model_name} on {self.device}...")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except:
            print("Fallback: Using trust_remote_code=True for tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
             
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Use float32 for stability
        dtype = torch.float32
        print("Using float32 for stability.")

        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype, 
                attn_implementation="eager" 
            )
        except:
            print("Fallback: Using trust_remote_code=True for model")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype, 
                trust_remote_code=True,
                attn_implementation="eager"
            )
        
        # Move to device
        print(f"Moving model to {self.device}...")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Verify device
        first_param_device = next(self.model.parameters()).device
        print(f"Model device verified: {first_param_device}")
        
        # Initialize MetaCLIP-H14 embedder for embeddings (native 1024D, no projection)
        print("Initializing MetaCLIP-H14 embedder for question embeddings...")
        self.embedder = MetaCLIPEmbedder(target_dim=None)  # Use native 1024D
        self.embedding_dim = self.embedder.get_dimension()
        print(f"MetaCLIP-H14 embedder initialized. Embedding dimension: {self.embedding_dim}")
        
        # Update model.config.hidden_size for compatibility with observation spaces
        class FakeConfig:
            def __init__(self, hidden_size):
                self.hidden_size = hidden_size
        original_config = self.model.config
        self.model.config = FakeConfig(self.embedding_dim)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using MetaCLIP-H14 embedder."""
        try:
            embedding = self.embedder.embed(text)
            if np.isnan(embedding).any():
                raise ValueError("NaNs in embedding")
            return embedding
        except Exception as e:
            print(f"Embedding failed: {e}")
            raise

    def _generate(self, prompt: str, active_tools: list = None, max_tokens: int = 512, prompt_suffix: str = None) -> str:
        """
        Core generation method with optional tool prompting and prompt modifiers.
        
        Args:
            prompt: The user prompt
            active_tools: List of tool names to enable
            max_tokens: Maximum tokens to generate
            prompt_suffix: Additional instructions to add to system prompt (from RL)
        """
        # Tool descriptions with usage examples
        TOOL_DESCRIPTIONS = {
            "calculator": (
                "calculator - Evaluate mathematical expressions.\n"
                "  Example: TOOL: calculator || QUERY: 125 * 4 + 89"
            ),
            "web_search": (
                "web_search - Search the web for current information.\n"
                "  Example: TOOL: web_search || QUERY: population of Tokyo 2024"
            ),
            # "python": (
            #     "python - Execute Python code. Use print() to output results. Has access to math module.\n"
            #     "  Example: TOOL: python || QUERY: print(sum([i**2 for i in range(1, 11)]))"
            # ),
            "python": (
                "python - Execute Python code. Use print() to output results. \n"
                "  Pre-imported libraries: pandas as pd, numpy as np, scipy, cv2, pdfplumber, PIL.Image, sklearn.\n"
                "  File access: You can read files using open() or pd.read_csv(), etc.\n"
                "  Example: TOOL: python || QUERY: df = pd.read_csv('file.csv'); print(df.head())"
            ),
            "ocr_reader": (
                "ocr_reader - Read and extract text from image files.\n"
                "  Example: TOOL: ocr_reader || QUERY: '/path/to/image.jpg'"
            )
        }
        
        # NOTE: Changed this prompt but maybe can be seperated with RL prompt modifiers.
        # Build system prompt
        # sys_prompt = "You are a helpful assistant that solves problems step by step."
        sys_prompt = (
            "You are an expert autonomous agent capable of solving complex tasks. "
            "Your goal is to provide the correct Final Answer.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. IF you are given a file path, you MUST use a tool (like 'python' or 'vision') to read it.\n"
            "2. Do NOT guess specific numbers or data from files you haven't read.\n"
            "3. Use Python for all complex calculations or data filtering.\n"
            "4. Format your final conclusion exactly as: Final Answer: <your_answer>"
        )
        
        # Add RL-selected prompt modifiers (if any)
        if prompt_suffix:
            sys_prompt += f" {prompt_suffix}"
        
        if active_tools:
            tools_text = "\n".join([TOOL_DESCRIPTIONS[t] for t in active_tools if t in TOOL_DESCRIPTIONS])
            sys_prompt += (
                f"\n\nYou have access to these tools:\n{tools_text}\n\n"
                "IMPORTANT: To use a tool, you MUST write EXACTLY this format: TOOL: <tool_name> || QUERY: <your_query>\n"
                "When you need to calculate, verify, or look up information, USE THE TOOLS. Do not try to solve everything yourself.\n"
                "For mathematical problems, use the calculator or python tool. For data lookups, use web_search."
            )
        
        # TODO: Check if we need this as it was not in zip file.
        else:
            sys_prompt += " Answer using your own knowledge only."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            try:
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            except Exception as e:
                print(f"Generation failed: {e}. Falling back to CPU...")
                self.model.to("cpu")
                inputs = inputs.to("cpu")
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                self.model.to(self.device)

        return self.tokenizer.decode(
            gen_ids[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )

    def reason(self, question: str, tools: list = None, tokens: int = 512, prompt_suffix: str = None) -> str:
        """Generate step-by-step reasoning for a question."""
        prompt = (
            f"Question: {question}\n"
            "Please break this down and think step-by-step to solve it. "
            "Do not just give the answer, show your work."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens, prompt_suffix=prompt_suffix)

    def verify(self, question: str, reasoning: str, tools: list = None, tokens: int = 256, prompt_suffix: str = None) -> str:
        """Verify and critique reasoning for a question."""
        prompt = (
            f"Question: {question}\n"
            f"Proposed Reasoning: {reasoning}\n"
            "Review this reasoning for logical errors or calculation mistakes. "
            "If it is correct, say 'The reasoning is correct.' "
            "If there is an error, point it out."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens, prompt_suffix=prompt_suffix)

    def answer_direct(self, question: str, tools: list = None, tokens: int = 128, prompt_suffix: str = None) -> str:
        """Generate a direct, concise answer."""
        prompt = f"Question: {question}\nAnswer concisely."
        return self._generate(prompt, active_tools=tools, max_tokens=tokens, prompt_suffix=prompt_suffix)

    def answer_with_context(self, question: str, context: str, tools: list = None, tokens: int = 128, prompt_suffix: str = None) -> str:
        """Generate an answer based on provided context/reasoning."""
        prompt = (
            f"Question: {question}\n"
            f"Context/Reasoning: {context}\n"
            "Based on the above, provide the final answer."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens, prompt_suffix=prompt_suffix)


class OpenRouterWorker:
    """OpenRouter API-based worker that handles reasoning, verification, and answering.
    Compatible interface with LLMWorker for drop-in replacement.
    Uses MetaCLIP-H14 for embeddings.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize OpenRouter worker.
        
        Args:
            model_name: OpenRouter model ID (e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet")
                       If None, uses OPENROUTER_MODEL from .env or defaults to "openai/gpt-4o"
            api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var)
        """
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Initialize MetaCLIP-H14 embedder for embeddings (native 1024D, no projection)
        print("Initializing MetaCLIP-H14 embedder for question embeddings...")
        self.embedder = MetaCLIPEmbedder(target_dim=None)  # Use native 1024D
        self.embedding_dim = self.embedder.get_dimension()
        print(f"MetaCLIP-H14 embedder initialized. Embedding dimension: {self.embedding_dim}")
        
        # Fake model.config for compatibility with existing code that expects model.config.hidden_size
        class FakeConfig:
            def __init__(self, hidden_size):
                self.hidden_size = hidden_size
        self.model = type('obj', (object,), {'config': FakeConfig(self.embedding_dim)})()
        
        print(f"OpenRouter Worker initialized with model: {self.model_name}")

    def _call_api(self, messages: list, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """
        Call OpenRouter API for text generation.
        
        Args:
            messages: List of message dicts with "role" and "content"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            
        Returns:
            Generated text response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/context-optimization",
            "X-Title": "Context Optimization RL"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload, 
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if "choices" not in result or len(result["choices"]) == 0:
                raise ValueError(f"Invalid API response: {result}")
            
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"OpenRouter API request error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"Error details: {error_detail}")
                except:
                    print(f"Response text: {e.response.text}")
            raise
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding using MetaCLIP-H14 embedder.
        
        Args:
            text: Input text
            
        Returns:
            numpy array embedding
        """
        try:
            embedding = self.embedder.embed(text)
            if np.isnan(embedding).any():
                raise ValueError("NaNs in embedding")
            return embedding
        except Exception as e:
            print(f"Embedding failed: {e}")
            raise

    def _generate(self, prompt: str, active_tools: list = None, max_tokens: int = 512, prompt_suffix: str = None) -> str:
        """
        Core generation method with optional tool prompting and prompt modifiers.
        Same interface as LLMWorker._generate for compatibility.
        
        Args:
            prompt: The user prompt
            active_tools: List of tool names to enable
            max_tokens: Maximum tokens to generate
            prompt_suffix: Additional instructions to add to system prompt (from RL)
        """
        # Tool descriptions with usage examples (same as LLMWorker)
        TOOL_DESCRIPTIONS = {
            "calculator": (
                "calculator - Evaluate mathematical expressions.\n"
                "  Example: TOOL: calculator || QUERY: 125 * 4 + 89"
            ),
            "web_search": (
                "web_search - Search the web for current information.\n"
                "  Example: TOOL: web_search || QUERY: population of Tokyo 2024"
            ),
            "python": (
                "python - Execute Python code. Use print() to output results. \n"
                "  Pre-imported libraries: pandas as pd, numpy as np, scipy, cv2, pdfplumber, PIL.Image, sklearn.\n"
                "  File access: You can read files using open() or pd.read_csv(), etc.\n"
                "  Example: TOOL: python || QUERY: df = pd.read_csv('file.csv'); print(df.head())"
            ),
            "ocr_reader": (
                "ocr_reader - Read and extract text from image files.\n"
                "  Example: TOOL: ocr_reader || QUERY: '/path/to/image.jpg'"
            )
        }
        
        # Build system prompt (same as LLMWorker)
        sys_prompt = (
            "You are an expert autonomous agent capable of solving complex tasks. "
            "Your goal is to provide the correct Final Answer.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. IF you are given a file path, you MUST use a tool (like 'python' or 'vision') to read it.\n"
            "2. Do NOT guess specific numbers or data from files you haven't read.\n"
            "3. Use Python for all complex calculations or data filtering.\n"
            "4. Format your final conclusion exactly as: Final Answer: <your_answer>"
        )
        
        # Add RL-selected prompt modifiers (if any)
        if prompt_suffix:
            sys_prompt += f" {prompt_suffix}"
        
        if active_tools:
            tools_text = "\n".join([TOOL_DESCRIPTIONS[t] for t in active_tools if t in TOOL_DESCRIPTIONS])
            sys_prompt += (
                f"\n\nYou have access to these tools:\n{tools_text}\n\n"
                "IMPORTANT: To use a tool, you MUST write EXACTLY this format: TOOL: <tool_name> || QUERY: <your_query>\n"
                "When you need to calculate, verify, or look up information, USE THE TOOLS. Do not try to solve everything yourself.\n"
                "For mathematical problems, use the calculator or python tool. For data lookups, use web_search."
            )
        else:
            sys_prompt += " Answer using your own knowledge only."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Call OpenRouter API
        return self._call_api(messages, max_tokens=max_tokens, temperature=0.0)

    def reason(self, question: str, tools: list = None, tokens: int = 512, prompt_suffix: str = None) -> str:
        """Generate step-by-step reasoning for a question."""
        prompt = (
            f"Question: {question}\n"
            "Please break this down and think step-by-step to solve it. "
            "Do not just give the answer, show your work."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens, prompt_suffix=prompt_suffix)

    def verify(self, question: str, reasoning: str, tools: list = None, tokens: int = 256, prompt_suffix: str = None) -> str:
        """Verify and critique reasoning for a question."""
        prompt = (
            f"Question: {question}\n"
            f"Proposed Reasoning: {reasoning}\n"
            "Review this reasoning for logical errors or calculation mistakes. "
            "If it is correct, say 'The reasoning is correct.' "
            "If there is an error, point it out."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens, prompt_suffix=prompt_suffix)

    def answer_direct(self, question: str, tools: list = None, tokens: int = 128, prompt_suffix: str = None) -> str:
        """Generate a direct, concise answer."""
        prompt = f"Question: {question}\nAnswer concisely."
        return self._generate(prompt, active_tools=tools, max_tokens=tokens, prompt_suffix=prompt_suffix)

    def answer_with_context(self, question: str, context: str, tools: list = None, tokens: int = 128, prompt_suffix: str = None) -> str:
        """Generate an answer based on provided context/reasoning."""
        prompt = (
            f"Question: {question}\n"
            f"Context/Reasoning: {context}\n"
            "Based on the above, provide the final answer."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens, prompt_suffix=prompt_suffix)

