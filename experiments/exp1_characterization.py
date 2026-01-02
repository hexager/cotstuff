"""
RQ1: Characterize when reasoning models use <think> tags.

Experiments:
- 1a: Usage patterns across difficulty and temperature
- 1b: Does model need the reasoning it produces?
"""
import pandas as pd
import numpy as np
from typing import List, Dict

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
    For samples that used <think>, regenerate without it.
    """
    print("\n=== Experiment 1b: Necessity Test ===")
    
    # Filter to only cases that used <think>
    think_cases = df_exp1a[df_exp1a['used_think'] == True].sample(
        n=min(50, len(df_exp1a[df_exp1a['used_think'] == True])),
        random_state=42
    )
    
    results = []
    
    for idx, row in think_cases.iterrows():
        # Original answer (with think)
        original_answer = row['answer']
        
        # Regenerate with temperature=0 (deterministic) and explicit instruction to NOT use <think>
        # This is a simplified test - you might want to be more sophisticated
        prompt_no_think = f"{row['prompt']}\n\nAnswer directly without showing your work:"
        
        no_think_result = model.generate_with_metadata(
            [prompt_no_think],
            temperature=0.0
        )[0]
        
        # Compare answers (simplified - you'd want better answer matching)
        answer_changed = original_answer.strip() != no_think_result['answer'].strip()
        
        results.append({
            'sample_id': row['sample_id'],
            'difficulty': row['difficulty'],
            'original_answer': original_answer,
            'no_think_answer': no_think_result['answer'],
            'answer_changed': answer_changed,
            'needed_think': answer_changed  # If answer changed, it needed the CoT
        })
    
    df_necessity = pd.DataFrame(results)
    
    print("\n=== Necessity Statistics ===")
    print(f"Cases where removing CoT changed answer: {df_necessity['needed_think'].sum()} / {len(df_necessity)}")
    print(f"Percentage: {df_necessity['needed_think'].mean() * 100:.1f}%")
    
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
