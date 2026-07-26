#!/usr/bin/env python3
"""Compare paged vs naive token-by-token on a short prompt."""
import re
import subprocess

N = 30


def run(args):
    res = subprocess.run(
        ["./build/mini_infer"] + args,
        capture_output=True, text=True, timeout=300,
    )
    body = res.stdout
    m = re.search(r"--- generated ---\n(.*?)\n--- end ---", body, re.DOTALL)
    text = m.group(1).strip() if m else ""
    m2 = re.search(r"(\d+) tokens in", body)
    n_new = int(m2.group(1)) if m2 else 0
    return text, n_new


# Use the same model with both paged and naive
model = "/data1/kdy/LLMs/Qwen2.5-Coder-1.5B-Instruct"
prompt = "你好"

print("Prompt:", repr(prompt))

print("\n[naive]")
text_naive, n_naive = run(
    ["--model", model, "--prompt", prompt, "--max-new-tokens", str(N),
     "--greedy", "--device", "0", "--max-seq-len", "1024"],
)
print(f"  n={n_naive}  text={text_naive!r}")

print("\n[paged]")
text_paged, n_paged = run(
    ["--model", model, "--prompt", prompt, "--max-new-tokens", str(N),
     "--greedy", "--device", "0", "--max-seq-len", "1024", "--paged"],
)
print(f"  n={n_paged}  text={text_paged!r}")

# Show diffs
print("\n--- comparison ---")
common = min(len(text_naive), len(text_paged))
for i in range(common):
    if text_naive[i] != text_paged[i]:
        print(f"first diff at char {i}: naive={text_naive[i]!r} paged={text_paged[i]!r}")
        break
print(f"naive len = {len(text_naive)}")
print(f"paged len = {len(text_paged)}")