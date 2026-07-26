#!/usr/bin/env python3
"""
Detailed comparison between mini-infer and HuggingFace transformers.
This script will help us identify where the forward pass diverges.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import json

def get_hf_intermediates(model_path, prompt, max_tokens=5):
    """Get intermediate values from HuggingFace model."""
    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to("cuda:0")
    model.eval()
    
    print(f"Encoding prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    input_ids = inputs['input_ids']
    
    print(f"Input tokens: {input_ids[0].tolist()}")
    print(f"Input length: {input_ids.shape[1]}")
    
    # Get embeddings
    with torch.no_grad():
        embeds = model.model.embed_tokens(input_ids)
        print(f"\nEmbeddings shape: {embeds.shape}")
        print(f"Embeddings mean: {embeds.mean().item():.6f}")
        print(f"Embeddings std: {embeds.std().item():.6f}")
        print(f"Embeddings first 10 values: {embeds[0, 0, :10].tolist()}")
    
    # Get hidden states after first layer
    with torch.no_grad():
        hidden_states = embeds
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        
        # Compute position embeddings (RoPE)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
        
        for i, layer in enumerate(model.model.layers[:1]):  # Just first layer
            # Input layernorm
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            print(f"\nAfter input_layernorm (layer {i}):")
            print(f"  Shape: {hidden_states.shape}")
            print(f"  Mean: {hidden_states.mean().item():.6f}")
            print(f"  Std: {hidden_states.std().item():.6f}")
            
            # Self attention
            attn_output = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_value=None,
                output_attentions=False,
                use_cache=False
            )[0]
            print(f"\nAfter self_attn (layer {i}):")
            print(f"  Shape: {attn_output.shape}")
            print(f"  Mean: {attn_output.mean().item():.6f}")
            print(f"  Std: {attn_output.std().item():.6f}")
            
            # Residual
            hidden_states = residual + attn_output
            print(f"\nAfter residual (layer {i}):")
            print(f"  Shape: {hidden_states.shape}")
            print(f"  Mean: {hidden_states.mean().item():.6f}")
            print(f"  Std: {hidden_states.std().item():.6f}")
            
            # Post-attention layernorm
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            print(f"\nAfter post_attention_layernorm (layer {i}):")
            print(f"  Shape: {hidden_states.shape}")
            print(f"  Mean: {hidden_states.mean().item():.6f}")
            print(f"  Std: {hidden_states.std().item():.6f}")
            
            # MLP
            mlp_output = layer.mlp(hidden_states)
            print(f"\nAfter MLP (layer {i}):")
            print(f"  Shape: {mlp_output.shape}")
            print(f"  Mean: {mlp_output.mean().item():.6f}")
            print(f"  Std: {mlp_output.std().item():.6f}")
            
            # Residual
            hidden_states = residual + mlp_output
            print(f"\nAfter residual (layer {i}):")
            print(f"  Shape: {hidden_states.shape}")
            print(f"  Mean: {hidden_states.mean().item():.6f}")
            print(f"  Std: {hidden_states.std().item():.6f}")
    
    # Get final logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        print(f"\nFinal logits shape: {logits.shape}")
        print(f"Logits mean: {logits.mean().item():.6f}")
        print(f"Logits std: {logits.std().item():.6f}")
        print(f"Logits max: {logits.max().item():.6f}")
        print(f"Logits min: {logits.min().item():.6f}")
        
        # Get top 5 tokens for last position
        last_logits = logits[0, -1, :]
        top5_indices = torch.topk(last_logits, 5).indices
        top5_tokens = [tokenizer.decode([idx]) for idx in top5_indices]
        print(f"\nTop 5 tokens for last position:")
        for i, (idx, token) in enumerate(zip(top5_indices, top5_tokens)):
            print(f"  {i+1}. {idx.item()}: '{token}' (logit: {last_logits[idx].item():.4f})")

def main():
    model_path = "/data1/kdy/LLMs/Qwen2.5-Coder-7B-Instruct"
    prompt = "Hello world"
    
    print("=" * 80)
    print("HuggingFace Intermediate Values")
    print("=" * 80)
    get_hf_intermediates(model_path, prompt)

if __name__ == "__main__":
    main()
