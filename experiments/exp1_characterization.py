"""
RQ1: Characterize when reasoning models use <think> tags.

Experiments:
- 1a: Usage patterns across difficulty and temperature
- 1b: Does model need the reasoning it produces?
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from src.data import GSM8KDataset
from src.model import ReasoningModel
from src.visualization import plot_usage_by_condition, plot_accuracy_comparison
from config import DATA_CONFIG, MODEL_CONFIG, PATH_CONFIG


def run_exp1a_usage_patterns(model: ReasoningModel, dataset: GSM8KDataset):
    """
    Generate completions across conditions and measure <think> usage.
    """
    print("\n=== Experiment 1a: Usage Patterns ===")
    
    # Get samples
    n_per_condition = DATA_CONFIG.n_characterization // 4  # Split across conditions
    easy_samples = dataset.get_samples(n_per_condition, difficulty="easy")
    hard_samples = dataset.get_samples(n_per_condition, difficulty="hard")
    
    all_results = []
    
    # Test each temperature
    for temp in MODEL_CONFIG.temperatures:
        print(f"\nTesting temperature: {temp}")
        
        # Easy questions
        easy_prompts = [s['question'] for s in easy_samples]
        easy_results = model.generate_with_metadata(easy_prompts, temperature=temp)
        for result, sample in zip(easy_results, easy_samples):
            result.update({
                'difficulty': 'easy',
                'sample_id': sample['id'],
                'ground_truth': sample['answer']
            })
        all_results.extend(easy_results)
        
        # Hard questions
        hard_prompts = [s['question'] for s in hard_samples]
        hard_results = model.generate_with_metadata(hard_prompts, temperature=temp)
        for result, sample in zip(hard_results, hard_samples):
            result.update({
                'difficulty': 'hard',
                'sample_id': sample['id'],
                'ground_truth': sample['answer']
            })
        all_results.extend(hard_results)
        
        model.clear_cache()
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Calculate statistics
    print("\n=== Usage Statistics ===")
    usage_by_condition = df.groupby(['difficulty', 'temperature'])['used_think'].agg([
        ('count', 'count'),
        ('used_think_pct', lambda x: x.mean() * 100),
        ('avg_think_length', lambda x: df.loc[x.index, 'think_length'].mean())
    ])
    print(usage_by_condition)
    
    # Save results
    df.to_csv(PATH_CONFIG.results_dir / "exp1a_raw_results.csv", index=False)
    usage_by_condition.to_csv(PATH_CONFIG.results_dir / "exp1a_usage_stats.csv")
    
    # Generate plots
    plot_usage_by_condition(df, PATH_CONFIG.figures_dir / "exp1a_usage_patterns.png")
    
    print(f"\n✓ Experiment 1a complete. Results saved to {PATH_CONFIG.results_dir}")
    
    return df


def run_exp1b_necessity_test(model: ReasoningModel, df_exp1a: pd.DataFrame):
    """
    Test if models need the CoT they produced.
    Compare answer WITH <think> vs WITHOUT <think> on same prompt.
    """
    print("\n=== Experiment 1b: Necessity Test ===")
    
    # Filter to only cases that used <think>
    think_cases = df_exp1a[df_exp1a['used_think'] == True]
    
    if len(think_cases) == 0:
        print("WARNING: No cases used <think> tags. Skipping Exp1b.")
        return pd.DataFrame()
    
    # Sample up to 50 cases
    n_sample = min(50, len(think_cases))
    think_cases = think_cases.sample(n=n_sample, random_state=42)
    
    results = []
    
    print(f"Testing {len(think_cases)} cases that originally used <think>...")
    
    for idx, row in tqdm(think_cases.iterrows(), total=len(think_cases), desc="Necessity test"):
        # Original answer (with think)
        original_answer = row['answer']
        original_prompt = row['prompt']
        
        # Regenerate SAME prompt multiple times at low temp to see consistency
        # If model consistently gives different answer, it "needed" the reasoning
        regenerations = model.generate_with_metadata(
            [original_prompt] * 3,  # Generate 3 times
            temperature=0.3  # Low but not zero
        )
        
        # Check consistency
        regen_answers = [r['answer'] for r in regenerations]
        regen_used_think = [r['used_think'] for r in regenerations]
        
        # Simple answer matching (you could make this more sophisticated)
        answers_consistent = len(set(regen_answers)) == 1
        still_uses_think = sum(regen_used_think) / len(regen_used_think)
        
        results.append({
            'sample_id': row['sample_id'],
            'difficulty': row['difficulty'],
            'original_temp': row['temperature'],
            'original_used_think': True,
            'original_answer': original_answer,
            'regen_think_rate': still_uses_think,
            'regen_answers_consistent': answers_consistent,
            'regen_answers': regen_answers
        })
    
    df_necessity = pd.DataFrame(results)
    
    print("\n=== Necessity Statistics ===")
    print(f"Cases that consistently use <think> on regen: {(df_necessity['regen_think_rate'] > 0.5).sum()} / {len(df_necessity)}")
    print(f"Cases with consistent answers: {df_necessity['regen_answers_consistent'].sum()} / {len(df_necessity)}")
    
    df_necessity.to_csv(PATH_CONFIG.results_dir / "exp1b_necessity_test.csv", index=False)
    
    print(f"\n✓ Experiment 1b complete. Results saved to {PATH_CONFIG.results_dir}")
    
    return df_necessity



def run_all_exp1():
    """Run all RQ1 experiments."""
    # Initialize
    dataset = GSM8KDataset()
    model = ReasoningModel()
    
    # Run experiments
    df_exp1a = run_exp1a_usage_patterns(model, dataset)
    df_exp1b = run_exp1b_necessity_test(model, df_exp1a)
    
    # Summary
    print("\n" + "="*50)
    print("RQ1 COMPLETE")
    print("="*50)
    print(f"Generated data for {len(df_exp1a)} samples")
    print(f"Tested necessity for {len(df_exp1b)} samples")
    print(f"\nOutputs in: {PATH_CONFIG.results_dir}")
    print(f"Figures in: {PATH_CONFIG.figures_dir}")
    
    return df_exp1a, df_exp1b


if __name__ == "__main__":
    run_all_exp1()
