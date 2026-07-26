#!/usr/bin/env python3
"""
Compare mini-infer output with HuggingFace transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_path = "/data1/kdy/LLMs/Qwen2.5-Coder-7B-Instruct"
    prompt = "Hello world"
    max_tokens = 20
    
    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to("cuda:0")
    model.eval()
    
    print(f"Encoding prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    print(f"Generating {max_tokens} tokens with greedy decoding...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0
        )
    
    # Decode only the generated part
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\nHuggingFace output:")
    print(f"Tokens: {generated_ids.tolist()}")
    print(f"Text: {generated_text}")
    
    # Also print the full sequence for debugging
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nFull sequence: {full_text}")

if __name__ == "__main__":
    main()
