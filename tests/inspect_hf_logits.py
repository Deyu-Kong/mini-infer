"""Compare mini-infer logits to HF logits on a single position to find any divergence."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/data1/kdy/LLMs/Qwen2.5-Coder-1.5B-Instruct"
prompt = "介绍一下你自己"

tok = AutoTokenizer.from_pretrained(model_path)
chat = f"user\n{prompt}\nassistant\n"
ids = tok.encode(chat, return_tensors="pt").to("cuda:0")

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda:0")
model.eval()

with torch.no_grad():
    out = model(ids)
logits = out.logits[0]  # [S, V]

print(f"Prompt len: {ids.shape[1]}")
print()
for pos in range(min(13, ids.shape[1])):
    top5 = torch.topk(logits[pos], 5)
    print(f"Position {pos}:")
    for v, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        tok_str = tok.decode([idx])
        print(f"  [{idx}] = {v:.4f}  ({tok_str!r})")
    print()