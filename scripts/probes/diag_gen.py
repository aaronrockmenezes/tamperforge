"""Does the panel's BATCHED generation differ from batch-1 on the attacked model?"""
import sys, torch
sys.path.insert(0,"src"); sys.path.insert(0,"experiments")
from tamperforge import load_model
from tamperforge.data import load_advbench_prompts
from version_a_eval_panel import generate
import os
os.environ["TF_QWEN_THINKING"] = "default"

m, tok, dev = load_model("outputs/xtest_v8_surg_k16")   # the EXPORTED attacked model
m.eval()
prompts = load_advbench_prompts(None, n=8, seed=42, source="walledai")

b1 = generate(m, tok, prompts, dev, 256, batch_size=1)
b8 = generate(m, tok, prompts, dev, 256, batch_size=8)
same = sum(a.strip() == b.strip() for a, b in zip(b1, b8))
print(f"identical outputs: {same}/{len(prompts)}\n")
for i, (a, b) in enumerate(zip(b1, b8)):
    if a.strip() != b.strip():
        print(f"--- prompt {i}: {prompts[i][:60]}")
        print(f"  batch1: {a[:200].replace(chr(10),' ')!r}")
        print(f"  batch8: {b[:200].replace(chr(10),' ')!r}")
        print()
        if i >= 2: break
