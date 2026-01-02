"""
All plotting and visualization functions.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_usage_by_condition(df: pd.DataFrame, output_path: Path):
    """Plot <think> usage by difficulty and temperature."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # By temperature
    usage_by_temp = df.groupby('temperature')['used_think'].mean() * 100
    axes[0].bar(usage_by_temp.index, usage_by_temp.values)
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('<think> Usage (%)')
    axes[0].set_title('Reasoning Engagement by Temperature')
    
    # By difficulty
    usage_by_diff = df.groupby('difficulty')['used_think'].mean() * 100
    axes[1].bar(usage_by_diff.index, usage_by_diff.values)
    axes[1].set_xlabel('Difficulty')
    axes[1].set_ylabel('<think> Usage (%)')
    axes[1].set_title('Reasoning Engagement by Difficulty')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output_path}")
    plt.close()


def plot_accuracy_comparison(df: pd.DataFrame, output_path: Path):
    """Compare accuracy with vs without <think>."""
    # This requires answer correctness evaluation
    # Placeholder for now
    pass
