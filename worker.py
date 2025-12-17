# worker.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
import numpy as np

class LLMWorker:
    def __init__(self):
        print(f"Loading Worker Model: {config.LLM_MODEL_NAME} on {config.DEVICE}...")
        # Try loading without trust_remote_code first if transformers is recent
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME) # trust_remote_code=True removed
        except:
             print("Fallback: Using trust_remote_code=True for tokenizer")
             self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME, trust_remote_code=True)
             
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # FIX: Determine the best dtype to avoid NaNs
        # Force float32 for maximum stability to debug "device-side assert"
        dtype = torch.float32
        print("Forcing float32 for stability.")
        
        # if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        #    dtype = torch.bfloat16  # Best for Ampere+ GPUs (3090, 4090, A100)
        #    print("Using bfloat16 for stability.")
        # else:
        #    dtype = torch.float32   # Slower but 100% stable on all GPUs
        #    print("Using float32 for stability.")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_NAME,
                torch_dtype=dtype, 
                # device_map="auto", # Removed to avoid accelerate locking device
                attn_implementation="eager" 
            )
        except:
             print("Fallback: Using trust_remote_code=True for model")
             self.model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_NAME,
                torch_dtype=dtype, 
                # device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"
            )
        self.model.to(config.DEVICE)
        self.model.eval()

    def get_embedding(self, text: str):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(config.DEVICE)
            with torch.no_grad():
                # Qwen2 model access might differ if not using AutoModel
                # Safest way is to run full model and get hidden states
                outputs = self.model(inputs.input_ids, output_hidden_states=True)
                # Last hidden state
                hidden_states = outputs.hidden_states[-1]
                embedding = hidden_states.mean(dim=1).squeeze().float().cpu().numpy()
                
                if np.isnan(embedding).any():
                     raise ValueError("NaNs in embedding on CUDA")
                return embedding
        except Exception as e:
            print(f"Embedding failed on {config.DEVICE}: {e}. Falling back to CPU...")
            self.model.to("cpu")
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cpu")
            with torch.no_grad():
                outputs = self.model(inputs.input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                return hidden_states.mean(dim=1).squeeze().float().cpu().numpy()

    def generate_response(self, question, active_tools, max_tokens):
        # Legacy method kept for compatibility if needed, but we will move to specific roles
        return self.answer_direct(question)

    # --- New Role-Based Generation Methods ---

    def _generate(self, prompt, active_tools=None, max_tokens=512):
        # 1. Build System Prompt with Tools
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
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
                print(f"Generation failed on {config.DEVICE}: {e}. Falling back to CPU...")
                self.model.to("cpu")
                inputs = inputs.to("cpu")
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                self.model.to(config.DEVICE) # Move back to GPU if possible

        return self.tokenizer.decode(gen_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    def reason(self, question, tools=None, tokens=512):
        prompt = (
            f"Question: {question}\n"
            "Please break this down and think step-by-step to solve it. "
            "Do not just give the answer, show your work."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens)

    def verify(self, question, reasoning, tools=None, tokens=256):
        prompt = (
            f"Question: {question}\n"
            f"Proposed Reasoning: {reasoning}\n"
            "Review this reasoning for logical errors or calculation mistakes. "
            "If it is correct, say 'The reasoning is correct.' "
            "If there is an error, point it out."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens)

    def answer_direct(self, question, tools=None, tokens=128):
        # Fast, direct answer
        prompt = f"Question: {question}\nAnswer concisely."
        return self._generate(prompt, active_tools=tools, max_tokens=tokens)

    def answer_with_context(self, question, context, tools=None, tokens=128):
        # Answer based on previous steps
        prompt = (
            f"Question: {question}\n"
            f"Context/Reasoning: {context}\n"
            "Based on the above, provide the final answer."
        )
        return self._generate(prompt, active_tools=tools, max_tokens=tokens)