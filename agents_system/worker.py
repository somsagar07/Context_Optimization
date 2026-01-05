import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import sys
import os
import requests
sys.path.append('..')
import config

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

class LLMWorker:
    """LLM-based worker that handles reasoning, verification, and answering."""
    
    def __init__(self, model_name: str = None):
        # Update the name if provided
        if model_name is not None:
            config.LLM_MODEL_NAME = model_name
        
        print(f"Loading Worker Model: {config.LLM_MODEL_NAME} on {config.DEVICE}...")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
        except:
            print("Fallback: Using trust_remote_code=True for tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL_NAME, 
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
                config.LLM_MODEL_NAME,
                torch_dtype=dtype, 
                attn_implementation="eager" 
            )
        except:
            print("Fallback: Using trust_remote_code=True for model")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_NAME,
                torch_dtype=dtype, 
                trust_remote_code=True,
                attn_implementation="eager"
            )
        
        # Move to device
        print(f"Moving model to {config.DEVICE}...")
        self.model = self.model.to(config.DEVICE)
        self.model.eval()
        
        # Verify device
        first_param_device = next(self.model.parameters()).device
        print(f"Model device verified: {first_param_device}")

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using model's hidden states."""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(config.DEVICE)
            
            with torch.no_grad():
                outputs = self.model(inputs.input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                embedding = hidden_states.mean(dim=1).squeeze().float().cpu().numpy()
                
                if np.isnan(embedding).any():
                    raise ValueError("NaNs in embedding")
                return embedding
                
        except Exception as e:
            print(f"Embedding failed on {config.DEVICE}: {e}. Falling back to CPU...")
            self.model.to("cpu")
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to("cpu")
            
            with torch.no_grad():
                outputs = self.model(inputs.input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                return hidden_states.mean(dim=1).squeeze().float().cpu().numpy()

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
                "To use a tool, write EXACTLY: TOOL: <tool_name> || QUERY: <your_query>\n"
                "Use tools to calculate, verify, or look up information when helpful."
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
        inputs = self.tokenizer([text], return_tensors="pt").to(config.DEVICE)
        
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
                self.model.to(config.DEVICE)

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
    Uses sentence-transformers for embeddings.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None, 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize OpenRouter worker.
        
        Args:
            model_name: OpenRouter model ID (e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet")
                       If None, uses OPENROUTER_MODEL from .env or defaults to "openai/gpt-4o"
            api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var)
            embedding_model: Sentence transformer model name for embeddings
        """
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Load sentence transformer for embeddings
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {embedding_model}...")
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Embedding dimension: {self.embedding_dim}")
        except ImportError:
            print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.embedding_model = None
            self.embedding_dim = 384  # Default fallback dimension
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embedding_model = None
            self.embedding_dim = 384
        
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
        Get text embedding using sentence transformers.
        
        Args:
            text: Input text
            
        Returns:
            numpy array embedding
        """
        if self.embedding_model is None:
            # Fallback: hash-based embedding
            import hashlib
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            # Create fixed-size embedding from hash
            embedding = np.frombuffer(hash_bytes[:self.embedding_dim*4], dtype=np.float32)[:self.embedding_dim]
            if len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)), mode='constant')
            return embedding
        
        # Use sentence transformer
        embedding = self.embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.astype(np.float32)

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
                "To use a tool, write EXACTLY: TOOL: <tool_name> || QUERY: <your_query>\n"
                "Use tools to calculate, verify, or look up information when helpful."
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

