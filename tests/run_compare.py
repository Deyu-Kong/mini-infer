#!/usr/bin/env python3
"""Side-by-side comparison of mini-infer and HuggingFace transformers outputs.

Runs the same prompt through both, with greedy decoding, and compares token
sequences.
"""
import re
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def hf_generate(model_path, prompt, max_new_tokens):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to("cuda:0")
    model.eval()

    chat = f"user\n{prompt}\nassistant\n"
    ids = tok.encode(chat, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.eos_token_id,
        )
    return out[0].tolist()


def mini_generate(model_path, prompt, max_new_tokens):
    cmd = [
        "./build/mini_infer",
        "--model", model_path,
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--greedy",
        "--device", "0",
        "--max-seq-len", "2048",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        print("STDERR:", res.stderr[-500:])
        raise RuntimeError(f"mini_infer failed rc={res.returncode}")
    txt = res.stdout
    m = re.search(r"--- generated ---\n(.*?)\n--- end ---", txt, re.DOTALL)
    body = m.group(1) if m else ""
    return body, txt


def main():
    model_path = "/data1/kdy/LLMs/Qwen2.5-Coder-1.5B-Instruct"
    prompt = "介绍一下你自己"
    n = 30

    print("=" * 70)
    print(f"Prompt: {prompt!r}   tokens={n}")
    print("=" * 70)

    print("\n[1] HuggingFace reference ...")
    hf_ids = hf_generate(model_path, prompt, n)
    tok = AutoTokenizer.from_pretrained(model_path)
    hf_text = tok.decode(hf_ids[len(tok.encode("user\n" + prompt + "\nassistant\n")):], skip_special_tokens=True)
    print("HF tokens :", hf_ids[len(tok.encode("user\n" + prompt + "\nassistant\n")):])
    print("HF text   :", hf_text)

    print("\n[2] mini-infer ...")
    mini_text, raw = mini_generate(model_path, prompt, n)
    print("mini text :", mini_text)

    match = "MATCH" if hf_text.strip() == mini_text.strip() else "DIFFER"
    print(f"\n>>> Result: {match}")
    if match != "MATCH":
        print("--- diff (char-level) ---")
        for i, (a, b) in enumerate(zip(hf_text, mini_text)):
            if a != b:
                print(f"  first diff at char {i}: HF={a!r}  mini={b!r}")
                break
        print(f"  HF len={len(hf_text)}  mini len={len(mini_text)}")


if __name__ == "__main__":
    main()