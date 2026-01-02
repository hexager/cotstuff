# test_deepseek.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

prompt = "Question: What is 23 + 47?\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)

completion = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\n" + "="*60)
print(completion)
print("="*60)
print(f"\nHas <think> tags: {'<think>' in completion}")
