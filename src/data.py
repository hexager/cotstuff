"""
Data loading and processing utilities.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from config import DATA_CONFIG, PATH_CONFIG


class GSM8KDataset:
    """Handles GSM8K dataset loading and processing."""
    
    def __init__(self):
        self.dataset = load_dataset(
            DATA_CONFIG.dataset_name,
            DATA_CONFIG.dataset_config,
            split=DATA_CONFIG.split
        )
        self.cache_file = PATH_CONFIG.data_dir / "gsm8k_processed.json"
    
    def get_samples(self, n_samples: int, difficulty: str = "mixed") -> List[Dict]:
        """
        Get n samples from dataset.
        
        Args:
            n_samples: Number of samples to return
            difficulty: "easy", "hard", or "mixed"
        
        Returns:
            List of dicts with 'question', 'answer', 'difficulty'
        """
        # For GSM8K, we'll use a simple heuristic:
        # Easy = shorter questions or fewer operations
        # Hard = longer questions or more operations
        
        samples = []
        for idx, item in enumerate(self.dataset):
            if len(samples) >= n_samples:
                break
            
            question = item['question']
            answer = item['answer']
            
            # Simple difficulty heuristic: count operations
            num_ops = question.count('+') + question.count('-') + \
                     question.count('*') + question.count('/')
            
            if difficulty == "easy" and num_ops <= 2:
                samples.append({
                    'id': idx,
                    'question': question,
                    'answer': answer,
                    'difficulty': 'easy'
                })
            elif difficulty == "hard" and num_ops > 2:
                samples.append({
                    'id': idx,
                    'question': question,
                    'answer': answer,
                    'difficulty': 'hard'
                })
            elif difficulty == "mixed":
                diff_label = 'easy' if num_ops <= 2 else 'hard'
                samples.append({
                    'id': idx,
                    'question': question,
                    'answer': answer,
                    'difficulty': diff_label
                })
        
        return samples
    
    def save_generations(self, generations: List[Dict], filename: str):
        """Save generation results to disk."""
        output_file = PATH_CONFIG.data_dir / filename
        with open(output_file, 'w') as f:
            json.dump(generations, f, indent=2)
        print(f"Saved {len(generations)} generations to {output_file}")
    
    def load_generations(self, filename: str) -> List[Dict]:
        """Load previously saved generations."""
        input_file = PATH_CONFIG.data_dir / filename
        if not input_file.exists():
            raise FileNotFoundError(f"No cached data at {input_file}")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} generations from {input_file}")
        return data
