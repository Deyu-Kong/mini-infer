#!/usr/bin/env python3
"""测试 Python 手工推理性能"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def test_python_inference():
    model_path = "/data1/kdy/LLMs/Qwen2.5-Coder-7B-Instruct"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16
    ).to("cuda:0")
    model.eval()
    
    prompt = "Hello"
    max_new_tokens = 50
    
    print(f"\nTesting with prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")
    
    # Warmup
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # Actual test
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    torch.cuda.synchronize()
    end = time.time()
    
    # Calculate metrics
    elapsed = end - start
    generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = generated_tokens / elapsed
    
    print(f"\nResults:")
    print(f"  Generated tokens: {generated_tokens}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {tokens_per_sec:.2f} tok/s")
    print(f"\nGenerated text:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    test_python_inference()
