import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel
import numpy as np
import sys
import os
import requests
import time
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
        # Preserve original config but update hidden_size to match embedding dimension
        # This ensures all transformers attributes (like is_encoder_decoder) are preserved
        original_config = self.model.config
        # Create a wrapper that preserves all original attributes but overrides hidden_size
        class ConfigWrapper:
            def __init__(self, original_config, hidden_size):
                self._original = original_config
                self.hidden_size = hidden_size
            
            def __getattr__(self, name):
                # Delegate to original config for any attribute not explicitly set
                return getattr(self._original, name)
        
        self.model.config = ConfigWrapper(original_config, self.embedding_dim)

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
        # Tool descriptions with usage examples - made more compelling and specific
        TOOL_DESCRIPTIONS = {
            "calculator": (
                "calculator - CRITICAL for ALL mathematical calculations!\n"
                "  USE THIS for: arithmetic, percentages, conversions, any math operations\n"
                "  DO NOT calculate manually - ALWAYS use this tool for math\n"
                "  Format: TOOL: calculator || QUERY: <expression>\n"
                "  Example: TOOL: calculator || QUERY: (125 * 4 + 89) / 3"
            ),
            "web_search": (
                "web_search - ESSENTIAL for factual questions and current information!\n"
                "  USE THIS for: ANY question about facts, names, dates, locations, statistics, current events, historical facts, scientific data, or ANY information that might not be in your training data\n"
                "  WHEN TO USE: If the question asks 'what', 'when', 'where', 'who', 'how many', 'which', or requires specific facts - USE web_search FIRST\n"
                "  DO NOT guess or rely on training data - web_search gives you ACCURATE, CURRENT information\n"
                "  Format: TOOL: web_search || QUERY: <your search query>\n"
                "  Example: TOOL: web_search || QUERY: population of Tokyo in 2024"
            ),
            "python": (
                "python - Powerful tool for data processing, file reading, and complex calculations!\n"
                "  USE THIS for: reading files (CSV, JSON, text), data analysis, complex calculations, image processing, data filtering/sorting\n"
                "  Pre-imported: pandas (pd), numpy (np), scipy, cv2, pdfplumber, PIL.Image, sklearn\n"
                "  File access: Use open(), pd.read_csv(), pd.read_json(), etc. to read files\n"
                "  Format: TOOL: python || QUERY: <your Python code with print()>\n"
                "  Example: TOOL: python || QUERY: df = pd.read_csv('data.csv'); print(df.describe())"
            ),
            "ocr_reader": (
                "ocr_reader - Extract text from images and scanned documents!\n"
                "  USE THIS for: reading text from images, PDFs, screenshots, scanned documents\n"
                "  WHEN TO USE: If you see an image file path or need to read text from an image\n"
                "  Format: TOOL: ocr_reader || QUERY: '<path/to/image.jpg>'\n"
                "  Example: TOOL: ocr_reader || QUERY: '/path/to/document.png'"
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
            
            # Stronger tool usage instructions, especially for web_search
            tool_usage_instructions = (
                "IMPORTANT: To use a tool, you MUST write EXACTLY this format: TOOL: <tool_name> || QUERY: <your_query>\n\n"
                "CRITICAL TOOL USAGE RULES:\n"
                "1. web_search - USE IT FIRST for factual questions!\n"
                "   - Questions starting with: 'What is', 'Who is', 'When did', 'Where is', 'How many', 'Which'\n"
                "   - Questions about: people, places, events, dates, statistics, facts, current information\n"
                "   - ANY question that asks for specific information - USE web_search BEFORE answering\n"
                "   - DO NOT guess - web_search gives you REAL, ACCURATE information\n"
                "   - Example: Question asks 'What is the capital of X?' → USE: TOOL: web_search || QUERY: capital of X\n"
                "2. calculator - USE IT for ALL math!\n"
                "   - Any arithmetic, percentages, calculations - USE calculator tool\n"
                "   - DO NOT calculate in your head or manually\n"
                "   - Example: Question asks 'What is 25% of 400?' → USE: TOOL: calculator || QUERY: 400 * 0.25\n"
                "3. python - USE IT for files and complex data!\n"
                "   - Reading files, data analysis, complex calculations\n"
                "   - Example: Question mentions a file → USE: TOOL: python || QUERY: df = pd.read_csv('file.csv'); print(...)\n"
                "4. REMEMBER: Tools are provided because you NEED them. If tools are available, you MUST use them.\n"
                "   - Not using available tools = WRONG answer\n"
                "   - Using tools = CORRECT answer\n"
            )
            
            sys_prompt += (
                f"\n\nYou have access to these tools:\n{tools_text}\n\n"
                f"{tool_usage_instructions}"
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

    def _call_api(self, messages: list, max_tokens: int = 512, temperature: float = 0.0, max_retries: int = 10) -> str:
        """
        Call OpenRouter API for text generation with retry logic and exponential backoff.
        
        Args:
            messages: List of message dicts with "role" and "content"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            max_retries: Maximum number of retry attempts (default: 10, increased for training stability)
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If all retry attempts fail
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
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=120  # 2 minute timeout
                )
                
                # Check status code before calling raise_for_status to handle edge cases
                status_code = response.status_code
                
                # Handle server errors (5xx) before raise_for_status
                if 500 <= status_code < 600:
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                        print(f"API server error {status_code} (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue  # Retry immediately
                    else:
                        last_exception = requests.exceptions.HTTPError(
                            f"{status_code} Server Error: {response.reason} for url: {response.url}",
                            response=response
                        )
                        print(f"API server error {status_code} after {max_retries} attempts")
                        break
                
                # Now raise for status to handle other HTTP errors
                response.raise_for_status()
                result = response.json()
                
                if "choices" not in result or len(result["choices"]) == 0:
                    raise ValueError(f"Invalid API response: {result}")
                
                choice = result["choices"][0]
                if "message" not in choice:
                    raise ValueError(f"Invalid API response: missing 'message' in choice: {choice}")
                
                message = choice["message"]
                if "content" not in message:
                    # Check if there's a finish_reason that might explain empty content
                    finish_reason = choice.get("finish_reason", "unknown")
                    error_msg = f"Invalid API response: missing 'content' in message. Finish reason: {finish_reason}. Full choice: {choice}"
                    print(f"WARNING: {error_msg}")
                    raise ValueError(error_msg)
                
                content = message["content"]
                if content is None:
                    finish_reason = choice.get("finish_reason", "unknown")
                    print(f"WARNING: API returned None content. Finish reason: {finish_reason}. Choice: {choice}")
                    return ""  # Return empty string instead of None
                
                return content
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                    print(f"API timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API timeout after {max_retries} attempts")
                    
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                    print(f"API connection error (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API connection error after {max_retries} attempts")
                    
            except requests.exceptions.HTTPError as e:
                # Check for rate limiting (429) or server errors (5xx)
                status_code = None
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                
                if status_code == 429:  # Rate limit
                    last_exception = e
                    if attempt < max_retries - 1:
                        # Try to get retry-after header, or use exponential backoff
                        retry_after = None
                        if hasattr(e, 'response') and e.response is not None:
                            retry_after = e.response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                wait_time = int(retry_after)
                            except (ValueError, TypeError):
                                wait_time = min(2 ** (attempt + 2), 120)
                        else:
                            wait_time = min(2 ** (attempt + 2), 120)  # Longer wait for rate limits
                        print(f"API rate limit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"API rate limit after {max_retries} attempts")
                elif status_code and 500 <= status_code < 600:  # Server errors (shouldn't reach here due to pre-check, but keep as fallback)
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 60)
                        print(f"API server error {status_code} (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"API server error {status_code} after {max_retries} attempts")
                else:
                    # Client errors (4xx except 429) - don't retry
                    print(f"API client error: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_detail = e.response.json()
                            print(f"Error details: {error_detail}")
                        except:
                            try:
                                print(f"Response text: {e.response.text[:500]}")  # Limit text length
                            except:
                                print(f"Response status: {e.response.status_code}")
                    raise
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 60)
                    print(f"API request error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API request error after {max_retries} attempts: {e}")
                    
            except ValueError as e:
                # Invalid response format - might be transient, retry a few times
                last_exception = e
                if attempt < max_retries - 1 and attempt < 2:  # Only retry twice for invalid responses
                    wait_time = 1
                    print(f"Invalid API response (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Invalid API response after retries: {e}")
                    raise
                    
            except Exception as e:
                # Unexpected errors - log and re-raise immediately
                print(f"Unexpected API error: {e}")
                raise
        
        # If we've exhausted all retries, raise the last exception
        if last_exception:
            print(f"API call failed after {max_retries} attempts. Last error: {last_exception}")
            raise last_exception
        else:
            raise Exception(f"API call failed after {max_retries} attempts with unknown error")

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
        # Tool descriptions with usage examples - made more compelling and specific (same as LLMWorker)
        TOOL_DESCRIPTIONS = {
            "calculator": (
                "calculator - CRITICAL for ALL mathematical calculations!\n"
                "  USE THIS for: arithmetic, percentages, conversions, any math operations\n"
                "  DO NOT calculate manually - ALWAYS use this tool for math\n"
                "  Format: TOOL: calculator || QUERY: <expression>\n"
                "  Example: TOOL: calculator || QUERY: (125 * 4 + 89) / 3"
            ),
            "web_search": (
                "web_search - ESSENTIAL for factual questions and current information!\n"
                "  USE THIS for: ANY question about facts, names, dates, locations, statistics, current events, historical facts, scientific data, or ANY information that might not be in your training data\n"
                "  WHEN TO USE: If the question asks 'what', 'when', 'where', 'who', 'how many', 'which', or requires specific facts - USE web_search FIRST\n"
                "  DO NOT guess or rely on training data - web_search gives you ACCURATE, CURRENT information\n"
                "  Format: TOOL: web_search || QUERY: <your search query>\n"
                "  Example: TOOL: web_search || QUERY: population of Tokyo in 2024"
            ),
            "python": (
                "python - Powerful tool for data processing, file reading, and complex calculations!\n"
                "  USE THIS for: reading files (CSV, JSON, text), data analysis, complex calculations, image processing, data filtering/sorting\n"
                "  Pre-imported: pandas (pd), numpy (np), scipy, cv2, pdfplumber, PIL.Image, sklearn\n"
                "  File access: Use open(), pd.read_csv(), pd.read_json(), etc. to read files\n"
                "  Format: TOOL: python || QUERY: <your Python code with print()>\n"
                "  Example: TOOL: python || QUERY: df = pd.read_csv('data.csv'); print(df.describe())"
            ),
            "ocr_reader": (
                "ocr_reader - Extract text from images and scanned documents!\n"
                "  USE THIS for: reading text from images, PDFs, screenshots, scanned documents\n"
                "  WHEN TO USE: If you see an image file path or need to read text from an image\n"
                "  Format: TOOL: ocr_reader || QUERY: '<path/to/image.jpg>'\n"
                "  Example: TOOL: ocr_reader || QUERY: '/path/to/document.png'"
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
            
            # Stronger tool usage instructions, especially for web_search
            tool_usage_instructions = (
                "IMPORTANT: To use a tool, you MUST write EXACTLY this format: TOOL: <tool_name> || QUERY: <your_query>\n\n"
                "CRITICAL TOOL USAGE RULES:\n"
                "1. web_search - USE IT FIRST for factual questions!\n"
                "   - Questions starting with: 'What is', 'Who is', 'When did', 'Where is', 'How many', 'Which'\n"
                "   - Questions about: people, places, events, dates, statistics, facts, current information\n"
                "   - ANY question that asks for specific information - USE web_search BEFORE answering\n"
                "   - DO NOT guess - web_search gives you REAL, ACCURATE information\n"
                "   - Example: Question asks 'What is the capital of X?' → USE: TOOL: web_search || QUERY: capital of X\n"
                "2. calculator - USE IT for ALL math!\n"
                "   - Any arithmetic, percentages, calculations - USE calculator tool\n"
                "   - DO NOT calculate in your head or manually\n"
                "   - Example: Question asks 'What is 25% of 400?' → USE: TOOL: calculator || QUERY: 400 * 0.25\n"
                "3. python - USE IT for files and complex data!\n"
                "   - Reading files, data analysis, complex calculations\n"
                "   - Example: Question mentions a file → USE: TOOL: python || QUERY: df = pd.read_csv('file.csv'); print(...)\n"
                "4. REMEMBER: Tools are provided because you NEED them. If tools are available, you MUST use them.\n"
                "   - Not using available tools = WRONG answer\n"
                "   - Using tools = CORRECT answer\n"
            )
            
            sys_prompt += (
                f"\n\nYou have access to these tools:\n{tools_text}\n\n"
                f"{tool_usage_instructions}"
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

