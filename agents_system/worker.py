import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import sys
sys.path.append('..')
import config

class LLMWorker:
    """LLM-based worker that handles reasoning, verification, and answering."""
    
    def __init__(self):
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

    def _generate(self, prompt: str, active_tools: list = None, max_tokens: int = 512) -> str:
        """Core generation method with optional tool prompting."""
        # Build system prompt
        sys_prompt = "You are a helpful assistant."
        if active_tools:
            sys_prompt += (
                f" You have access to these tools: {active_tools}. "
                "If you need to use one, output EXACTLY in this format: "
                "TOOL: <tool_name> || QUERY: <query>. "
                "Otherwise, just answer the question."
            )
        else:
            sys_prompt += " You have NO external tools. Answer using your own knowledge."

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

    def reason(self, question: str, tools: list = None, tokens: int = 512) -> str:
        """Generate step-by-step reasoning for a question."""
        prompt = (
            f"Question: {question}\n"
            "Please break this down and think step-by-step to solve it. "
            "Do not just give the answer, show your work."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens)

    def verify(self, question: str, reasoning: str, tools: list = None, tokens: int = 256) -> str:
        """Verify and critique reasoning for a question."""
        prompt = (
            f"Question: {question}\n"
            f"Proposed Reasoning: {reasoning}\n"
            "Review this reasoning for logical errors or calculation mistakes. "
            "If it is correct, say 'The reasoning is correct.' "
            "If there is an error, point it out."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens)

    def answer_direct(self, question: str, tools: list = None, tokens: int = 128) -> str:
        """Generate a direct, concise answer."""
        prompt = f"Question: {question}\nAnswer concisely."
        return self._generate(prompt, active_tools=tools, max_tokens=tokens)

    def answer_with_context(self, question: str, context: str, tools: list = None, tokens: int = 128) -> str:
        """Generate an answer based on provided context/reasoning."""
        prompt = (
            f"Question: {question}\n"
            f"Context/Reasoning: {context}\n"
            "Based on the above, provide the final answer."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens)

