# final_model_test.py - Run this RIGHT NOW
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

models_to_test = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen2.5-Math-7B-Instruct",
]

for model_name in models_to_test:
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print('='*70)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        question = "Question: What is 127 * 83? Show your work step by step."
        inputs = tokenizer(question, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.9)
        result = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        print(f"✓ Loaded successfully")
        print(f"Output:\n{result}")
        print(f"Has <think>: {'<think>' in result.lower()}")
        print(f"Shows reasoning: {'*' in result or 'multiply' in result.lower()}")
        
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ Failed: {e}")
