# test_setup.py
from src.data import GSM8KDataset
from src.model import ReasoningModel

dataset = GSM8KDataset()
samples = dataset.get_samples(5, difficulty="mixed")
print(f"Loaded {len(samples)} samples")

model = ReasoningModel()
prompts = [s['question'] for s in samples[:2]]
results = model.generate_with_metadata(prompts, temperature=0.7)
print(f"Generated {len(results)} completions")
print(f"Used <think>: {sum(r['used_think'] for r in results)}")
