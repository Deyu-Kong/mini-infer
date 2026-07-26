#!/usr/bin/env python3
"""Generate a synthetic ShareGPT-style dataset (1000 prompts).

Writes a JSON file at `benchmarks/datasets/sharegpt_sample.json`. Each entry has:
  {
    "conversations": [
      {"from": "human", "value": "<prompt text>"},
      {"from": "gpt",   "value": "<placeholder>"}
    ]
  }

Length distribution matches what we typically see in real ShareGPT:
  * 50%  short  (32-128 chars)
  * 30%  medium (256-1024 chars)
  * 20%  long   (2048-4096 chars)
"""
import json
import os
import random
import sys

random.seed(2026)

TEMPLATES = [
    # short
    "Hi!",
    "Translate 'good morning' to Spanish.",
    "What is the capital of France?",
    "Write a haiku about a cat.",
    "Explain what a GPU does.",
    "List three uses of Python.",
    "Define the word 'serendipity'.",
    "How do I boil an egg?",
    "Continue: The rain in Spain",
    "Summarize: Einstein was a physicist.",
    # medium
    "Explain the difference between supervised and unsupervised learning, "
    "giving one concrete example of each.",
    "You are a senior backend engineer. Review the following API design and "
    "list three issues with it. Response format: numbered list.",
    "Write a Python function that takes a list of integers and returns the "
    "longest strictly increasing subsequence. Include docstring and a unit "
    "test using pytest.",
    "Translate the following passage from English to Mandarin Chinese, "
    "preserving technical vocabulary.",
    "Draft a 200-word product description for a new noise-cancelling "
    "headphone, suitable for an e-commerce website.",
    "Read the following short story and identify the main conflict.",
    "Compare TCP and UDP at the transport layer. Mention at least three "
    "differences and one scenario where each is preferred.",
    "Explain how transformers use self-attention and why it scales better "
    "than RNNs for long sequences.",
    "Give me a meal plan for a vegetarian athlete training for a marathon, "
    "covering breakfast, lunch, dinner, and one snack for 7 days.",
    "Write a SQL query that returns the top 10 customers by total spend in "
    "the last 90 days, joining the orders and customers tables.",
    # long
    "Write a 500-word essay on the impact of open-source large language "
    "models on academic research. Your essay should have an introduction, "
    "two body paragraphs, and a conclusion. Use specific examples.",
    "You are a senior staff engineer preparing a system design document. "
    "Outline a design for a real-time chat system that supports 10M daily "
    "active users, including load balancing, message persistence, presence, "
    "and end-to-end encryption. Be specific about technology choices and "
    "trade-offs.",
    "Read the following legal contract clause and produce a plain-English "
    "summary of the obligations of each party, the duration, and any "
    "termination conditions. Clause: " + ("lorem ipsum " * 80),
    "Design a comprehensive test plan for an in-vehicle infotainment "
    "system. Cover functional tests, performance benchmarks, security "
    "audits, and regulatory compliance. Include a risk matrix and a "
    "go/no-go checklist for the release.",
]

def make_text(target_chars: int) -> str:
    out = []
    length = 0
    while length < target_chars:
        t = random.choice(TEMPLATES)
        out.append(t)
        length += len(t) + 1
    return " ".join(out)[:target_chars]

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    out_path = os.path.join(os.path.dirname(__file__),
                            "sharegpt_sample.json")
    prompts = []
    buckets = [
        (32, 128, 50),     # short
        (256, 1024, 30),   # medium
        (2048, 4096, 20),  # long
    ]
    weights = [w for _, _, w in buckets]
    dist = random.choices(range(len(buckets)), weights=weights, k=n)
    for i in range(n):
        lo, hi, _ = buckets[dist[i]]
        text = make_text(random.randint(lo, hi))
        prompts.append({
            "conversations": [
                {"from": "human", "value": text},
                {"from": "gpt",   "value": "(placeholder)"},
            ]
        })
    with open(out_path, "w") as f:
        json.dump(prompts, f)
    print(f"Wrote {n} prompts to {out_path}")

if __name__ == "__main__":
    main()