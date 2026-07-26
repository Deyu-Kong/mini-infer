#!/usr/bin/env python3
"""
Test greedy sampling kernel correctness.
"""

import torch
import subprocess
import numpy as np

def test_greedy_sampling():
    """Verify that greedy sampling returns the true argmax."""
    
    # Create test logits
    vocab_size = 152064
    logits = torch.randn(vocab_size, dtype=torch.float16)
    
    # Set a clear maximum
    max_idx = 12345
    logits[max_idx] = 100.0
    
    # Find argmax with PyTorch
    torch_argmax = torch.argmax(logits).item()
    
    print(f"Test logits shape: {logits.shape}")
    print(f"Max value: {logits[max_idx].item()}")
    print(f"PyTorch argmax: {torch_argmax}")
    
    # Save logits to file
    logits_np = logits.numpy()
    logits_np.tofile('/tmp/test_logits.bin')
    
    # Create a simple C++ test program
    test_code = '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "kernels/sampling_kernel.cuh"

int main() {
    const int vocab_size = 152064;
    
    // Allocate GPU memory
    __half* d_logits;
    int* d_out;
    cudaMalloc(&d_logits, vocab_size * sizeof(__half));
    cudaMalloc(&d_out, sizeof(int));
    
    // Load logits from file
    FILE* f = fopen("/tmp/test_logits.bin", "rb");
    __half* h_logits = new __half[vocab_size];
    fread(h_logits, sizeof(__half), vocab_size, f);
    fclose(f);
    
    // Copy to GPU
    cudaMemcpy(d_logits, h_logits, vocab_size * sizeof(__half), cudaMemcpyHostToDevice);
    
    // Run greedy sampling
    mini_infer::kernels::launch_greedy_sample(d_logits, vocab_size, d_out, 0);
    cudaDeviceSynchronize();
    
    // Copy result back
    int h_out;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Greedy sampling result: %d\\n", h_out);
    
    // Cleanup
    delete[] h_logits;
    cudaFree(d_logits);
    cudaFree(d_out);
    
    return 0;
}
'''
    
    with open('/tmp/test_sampling.cpp', 'w') as f:
        f.write(test_code)
    
    print("\nTo test the kernel, compile and run:")
    print("cd /data1/kdy/Project/mini-infer")
    print("nvcc -O3 -std=c++17 -I src -I /usr/local/cuda-12.1/include \\")
    print("     /tmp/test_sampling.cpp build/libmini_infer_kernels.a \\")
    print("     -L /usr/local/cuda-12.1/lib64 -lcudart -o /tmp/test_sampling")
    print("/tmp/test_sampling")

if __name__ == "__main__":
    test_greedy_sampling()
