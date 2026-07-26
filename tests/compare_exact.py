#!/usr/bin/env python3
"""
Exact comparison between mini-infer and HuggingFace.
Uses the SAME input tokens and compares intermediate values.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import json

def get_hf_intermediates(model_path, input_ids):
    """Get intermediate values from HuggingFace model."""
    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to("cuda:0")
    model.eval()
    
    print("Input tokens:", input_ids.tolist())
    print("Input length:", input_ids.shape[1])
    
    # Get embeddings
    with torch.no_grad():
        embeds = model.model.embed_tokens(input_ids)
        print("\n=== Embeddings ===")
        print("Shape:", embeds.shape)
        print("Mean:", embeds.mean().item())
        print("Std:", embeds.std().item())
        print("First 10 values:", embeds[0, 0, :10].tolist())
    
    # Get hidden states after first layer
    with torch.no_grad():
        hidden_states = embeds
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        
        # Compute position embeddings (RoPE)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
        
        for i, layer in enumerate(model.model.layers):  # Check all layers
            if i in [0, 3, 6, 7, 8, 9, 10, 15, 20, 27]:
                print("\n=== Layer", i, "===")
            
            # Input layernorm
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            print("\nAfter input_layernorm:")
            print("  Shape:", hidden_states.shape)
            print("  Mean:", hidden_states.mean().item())
            print("  Std:", hidden_states.std().item())
            print("  First 10:", hidden_states[0, 0, :10].tolist())
            
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
            print("\nAfter self_attn:")
            print("  Shape:", attn_output.shape)
            print("  Mean:", attn_output.mean().item())
            print("  Std:", attn_output.std().item())
            print("  First 10:", attn_output[0, 0, :10].tolist())
            
            # Residual
            hidden_states = residual + attn_output
            print("\nAfter residual:")
            print("  Shape:", hidden_states.shape)
            print("  Mean:", hidden_states.mean().item())
            print("  Std:", hidden_states.std().item())
            print("  First 10:", hidden_states[0, 0, :10].tolist())
            
            # Post-attention layernorm
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            print("\nAfter post_attention_layernorm:")
            print("  Shape:", hidden_states.shape)
            print("  Mean:", hidden_states.mean().item())
            print("  Std:", hidden_states.std().item())
            print("  First 10:", hidden_states[0, 0, :10].tolist())
            
            # MLP - check intermediate values
            gate = layer.mlp.gate_proj(hidden_states)
            up = layer.mlp.up_proj(hidden_states)
            silu = torch.nn.functional.silu(gate)
            mlp_intermediate = silu * up
            mlp_output = layer.mlp.down_proj(mlp_intermediate)
            
            print("\nMLP gate:")
            print("  Shape:", gate.shape)
            print("  Mean:", gate.mean().item())
            print("  Std:", gate.std().item())
            print("  First 10:", gate[0, 0, :10].tolist())
            
            print("\nMLP up:")
            print("  Shape:", up.shape)
            print("  Mean:", up.mean().item())
            print("  Std:", up.std().item())
            print("  First 10:", up[0, 0, :10].tolist())
            
            print("\nMLP silu:")
            print("  Shape:", silu.shape)
            print("  Mean:", silu.mean().item())
            print("  Std:", silu.std().item())
            print("  First 10:", silu[0, 0, :10].tolist())
            
            print("\nMLP intermediate (silu * up):")
            print("  Shape:", mlp_intermediate.shape)
            print("  Mean:", mlp_intermediate.mean().item())
            print("  Std:", mlp_intermediate.std().item())
            print("  First 10:", mlp_intermediate[0, 0, :10].tolist())
            
            print("\nAfter MLP:")
            print("  Shape:", mlp_output.shape)
            print("  Mean:", mlp_output.mean().item())
            print("  Std:", mlp_output.std().item())
            print("  First 10:", mlp_output[0, 0, :10].tolist())
            
            # Residual
            hidden_states = residual + mlp_output
            print("\nAfter residual:")
            print("  Shape:", hidden_states.shape)
            print("  Mean:", hidden_states.mean().item())
            print("  Std:", hidden_states.std().item())
            print("  First 10:", hidden_states[0, 0, :10].tolist())
    
    # Get final logits
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        logits = outputs.logits
        
        # Get final hidden states (before final norm)
        final_hidden = outputs.hidden_states[-1]
        print("\n=== Final Hidden States (before final norm) ===")
        print("Shape:", final_hidden.shape)
        print("Mean:", final_hidden.mean().item())
        print("Std:", final_hidden.std().item())
        print("First 10 values:", final_hidden[0, 0, :10].tolist())
        
        # Get final hidden states (after final norm)
        final_hidden_normed = model.model.norm(final_hidden)
        print("\n=== Final Hidden States (after final norm) ===")
        print("Shape:", final_hidden_normed.shape)
        print("Mean:", final_hidden_normed.mean().item())
        print("Std:", final_hidden_normed.std().item())
        print("First 10 values:", final_hidden_normed[0, 0, :10].tolist())
        
        print("\n=== Final Logits ===")
        print("Shape:", logits.shape)
        print("Mean:", logits.mean().item())
        print("Std:", logits.std().item())
        print("Max:", logits.max().item())
        print("Min:", logits.min().item())
        
        # Get top 5 tokens for last position
        last_logits = logits[0, -1, :]
        top5_indices = torch.topk(last_logits, 5).indices
        top5_tokens = [tokenizer.decode([idx]) for idx in top5_indices]
        print("\nTop 5 tokens for last position:")
        for i, (idx, token) in enumerate(zip(top5_indices, top5_tokens)):
            print("  {}. {}: '{}' (logit: {:.4f})".format(
                i+1, idx.item(), token, last_logits[idx].item()))

def main():
    model_path = "/data1/kdy/LLMs/Qwen2.5-Coder-7B-Instruct"
    
    # Use ChatML format (same as mini-infer)
    prompt = "Hello"
    chat = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    
    print("=" * 80)
    print("HuggingFace Intermediate Values (ChatML format)")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids = tokenizer.encode(chat, return_tensors="pt").to("cuda:0")
    
    get_hf_intermediates(model_path, input_ids)

if __name__ == "__main__":
    main()
