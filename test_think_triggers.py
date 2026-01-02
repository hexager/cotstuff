# test_think_triggers.py
from src.model import ReasoningModel

model = ReasoningModel()

# Try explicit prompting
test_prompts = [
    # Baseline
    "Question: What is 23 * 47?",
    
    # With reasoning instruction
    "Question: What is 23 * 47?\nLet's think step by step:",
    
    # With DeepSeek format
    "User: What is 23 * 47?\n\nAssistant:",
    
    # Multi-step problem
    "Question: A store has 156 apples. They sell 47 in the morning and 38 in the afternoon. Then they receive a delivery of 89 apples. How many apples do they have now?",
    
    # Word problem
    "Question: If a train travels at 65 mph for 3.5 hours, then slows to 45 mph for 2 hours, what is the total distance traveled?",
]

results = model.generate_with_metadata(test_prompts, temperature=0.9)

for i, (prompt, result) in enumerate(zip(test_prompts, results), 1):
    print(f"\n{'='*70}")
    print(f"TEST {i}")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Used <think>: {result['used_think']}")
    print(f"Think length: {result['think_length']}")
    if result['used_think']:
        print(f"Think preview: {result['think_content'][:100]}...")
    else:
        print(f"Answer: {result['answer'][:200]}...")
    print('='*70)
