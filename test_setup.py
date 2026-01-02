# test_setup.py
from src.data import GSM8KDataset
from src.model import ReasoningModel

dataset = GSM8KDataset()
samples = dataset.get_samples(2, difficulty="mixed")
print(f"Loaded {len(samples)} samples\n")

model = ReasoningModel()
prompts = [s['question'] for s in samples[:2]]
results = model.generate_with_metadata(prompts, temperature=0.7)

print("="*60)
for i, result in enumerate(results):
    print(f"\nSAMPLE {i+1}:")
    print(f"PROMPT: {result['prompt'][:100]}...")
    print(f"\nFULL COMPLETION:\n{result['completion']}")
    print(f"\nUsed <think>: {result['used_think']}")
    print("="*60)
