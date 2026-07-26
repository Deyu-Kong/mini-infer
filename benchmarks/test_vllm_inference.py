#!/usr/bin/env python3
"""测试 vLLM 推理性能"""

from vllm import LLM, SamplingParams
import time

def test_vllm_inference():
    model_path = "/data1/kdy/LLMs/Qwen2.5-Coder-7B-Instruct"
    
    print("Loading model with vLLM...")
    llm = LLM(
        model=model_path,
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=2048
    )
    
    prompt = "Hello"
    max_new_tokens = 50
    
    print(f"\nTesting with prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")
    
    # Warmup
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10
    )
    _ = llm.generate([prompt], sampling_params)
    
    # Actual test
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens
    )
    
    start = time.time()
    outputs = llm.generate([prompt], sampling_params)
    end = time.time()
    
    # Calculate metrics
    elapsed = end - start
    generated_tokens = len(outputs[0].outputs[0].token_ids)
    tokens_per_sec = generated_tokens / elapsed
    
    print(f"\nResults:")
    print(f"  Generated tokens: {generated_tokens}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {tokens_per_sec:.2f} tok/s")
    print(f"\nGenerated text:")
    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    test_vllm_inference()
