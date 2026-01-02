"""
Model loading and generation utilities (raw transformers version).
"""
import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from config import MODEL_CONFIG


class ReasoningModel:
    """Wrapper for reasoning model with generation utilities."""
    
    def __init__(self):
        print(f"Loading model: {MODEL_CONFIG.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG.model_name,
            torch_dtype=getattr(torch, MODEL_CONFIG.dtype),
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        print(f"Model loaded. Vocab size: {len(self.tokenizer)}")
    
    def generate_with_metadata(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        batch_size: int = 4
    ) -> List[Dict]:
        """Generate completions and extract metadata."""
        results = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
            
            # Handle temperature=0 case (greedy decoding)
            if temperature == 0.0 or temperature < 0.01:
                gen_kwargs = {
                    **inputs,
                    'max_new_tokens': MODEL_CONFIG.max_new_tokens,
                    'do_sample': False,  # Greedy decoding
                    'pad_token_id': self.tokenizer.pad_token_id
                }
            else:
                gen_kwargs = {
                    **inputs,
                    'max_new_tokens': MODEL_CONFIG.max_new_tokens,
                    'temperature': temperature,
                    'top_p': MODEL_CONFIG.top_p,
                    'do_sample': True,
                    'pad_token_id': self.tokenizer.pad_token_id
                }
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)
            
            # Decode
            for prompt, output_ids in zip(batch_prompts, outputs):
                # Decode only the new tokens
                prompt_length = len(self.tokenizer.encode(prompt))
                completion_ids = output_ids[prompt_length:]
                completion = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
                
                metadata = self._extract_metadata(completion)
                
                results.append({
                    'prompt': prompt,
                    'completion': completion,
                    'temperature': temperature,
                    **metadata
                })
        
        return results

    
    def _extract_metadata(self, completion: str) -> Dict:
        """Extract <think> tag usage and content from completion."""
        has_think_start = "<think>" in completion
        has_think_end = "</think>" in completion
        used_think = has_think_start and has_think_end
        
        if used_think:
            start_idx = completion.find("<think>") + len("<think>")
            end_idx = completion.find("</think>")
            think_content = completion[start_idx:end_idx].strip()
            answer = completion[end_idx + len("</think>"):].strip()
            think_length = len(think_content.split())
        else:
            think_content = ""
            answer = completion.strip()
            think_length = 0
        
        return {
            'used_think': used_think,
            'think_content': think_content,
            'answer': answer,
            'think_length': think_length
        }
    
    def get_activations(
        self,
        text: str,
        layers: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Get activations at specified layers.
        Returns dict: {layer_idx: activation_tensor}
        """
        activations = {}
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                # output is typically (batch, seq, hidden_dim)
                activations[layer_idx] = output[0].detach().cpu()
            return hook
        
        # Register hooks
        hooks = []
        if layers is None:
            layers = range(len(self.model.model.layers))
        
        for layer_idx in layers:
            hook = self.model.model.layers[layer_idx].register_forward_hook(
                hook_fn(layer_idx)
            )
            hooks.append(hook)
        
        # Forward pass
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
