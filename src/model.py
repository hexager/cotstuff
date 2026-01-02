"""
Model loading and generation utilities.
"""
import torch
from typing import List, Dict, Optional
from transformer_lens import HookedTransformer
from tqdm import tqdm

from config import MODEL_CONFIG


class ReasoningModel:
    """Wrapper for reasoning model with generation utilities."""
    
    def __init__(self):
        print(f"Loading model: {MODEL_CONFIG.model_name}")
        self.model = HookedTransformer.from_pretrained(
            MODEL_CONFIG.model_name,
            device=MODEL_CONFIG.device,
            dtype=getattr(torch, MODEL_CONFIG.dtype)
        )
        self.tokenizer = self.model.tokenizer
        print(f"Model loaded. Vocabulary size: {self.model.cfg.d_vocab}")
    
    def generate_with_metadata(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        batch_size: int = 4
    ) -> List[Dict]:
        """
        Generate completions and extract metadata.
        
        Returns:
            List of dicts with 'prompt', 'completion', 'used_think',
            'think_content', 'answer', 'think_length'
        """
        results = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i+batch_size]
            
            # Generate
            outputs = self.model.generate(
                batch_prompts,
                max_new_tokens=MODEL_CONFIG.max_new_tokens,
                temperature=temperature,
                top_p=MODEL_CONFIG.top_p,
                do_sample=True
            )
            
            # Process each output
            for prompt, output in zip(batch_prompts, outputs):
                completion = output[len(prompt):]
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
            # Extract content between tags
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
    
    def get_logits_at_position(
        self,
        text: str,
        position: int = -1
    ) -> torch.Tensor:
        """Get logits at a specific token position."""
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(MODEL_CONFIG.device)
        with torch.no_grad():
            logits = self.model(tokens)
        return logits[0, position, :]
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
