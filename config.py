"""
Central configuration for all experiments.
Modify parameters here, not in individual scripts.
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    """Model and generation settings."""
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device: str = "cuda"
    dtype: str = "float16"  # Memory optimization for Kaggle
    max_new_tokens: int = 512
    
    # Generation parameters for characterization
    temperatures: list = (0.6, 0.7, 0.8)
    top_p: float = 0.95


@dataclass
class DataConfig:
    """Dataset and sampling settings."""
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    split: str = "test"
    
    # Sampling for experiments
    n_characterization: int = 200  # RQ1
    n_mechanism: int = 100         # RQ2 (50 per condition)
    n_steering: int = 50           # RQ3
    
    easy_difficulty_range: tuple = (1, 2)  # Define based on GSM8K structure
    hard_difficulty_range: tuple = (3, 5)


@dataclass
class ExperimentConfig:
    """Experiment-specific settings."""
    # RQ2: Mechanism investigation
    layers_to_analyze: list = None  # None = all layers
    attention_heads: int = None     # None = all heads
    
    # RQ3: Steering
    intervention_strengths: list = (-0.1, -0.05, 0.0, 0.05, 0.1)
    intervention_layers: list = None  # Will be determined from RQ2
    
    # Analysis
    random_seed: int = 42


@dataclass
class PathConfig:
    """File paths for outputs."""
    project_root: Path = Path(__file__).parent
    output_dir: Path = project_root / "outputs"
    data_dir: Path = output_dir / "data"
    figures_dir: Path = output_dir / "figures"
    results_dir: Path = output_dir / "results"
    
    def create_dirs(self):
        """Create all output directories."""
        for dir_path in [self.data_dir, self.figures_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global config instances
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
EXP_CONFIG = ExperimentConfig()
PATH_CONFIG = PathConfig()

# Initialize directories
PATH_CONFIG.create_dirs()
