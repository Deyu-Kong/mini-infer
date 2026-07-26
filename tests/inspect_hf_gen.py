"""Print HF top-5 logits for each generation step."""
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
logits = out.logits[0]
print("logits shape:", logits.shape)

step_logits_full = out.logits
cur = ids
for step in range(20):
    pos = step_logits_full.shape[1] - 1
    last = step_logits_full[0, pos]
    print(f"\nGen step {step}  pos={pos}  last.shape={tuple(last.shape)}")
    top5 = torch.topk(last, k=5)
    for v, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  [{idx}] = {v:.4f}  ({tok.decode([idx])!r})")
    nxt = top5.indices[0].view(1, 1)
    cur = torch.cat([cur, nxt.to(cur.device)], dim=1)
    with torch.no_grad():
        step_logits_full = model(cur).logits

print("\nfinal ids:", cur[0].tolist())