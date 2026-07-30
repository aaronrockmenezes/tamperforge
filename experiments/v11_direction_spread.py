"""How wide is v8's training direction ensemble, in degrees, vs the surgical rotation?

Reproduces the exact direction sampling in train_tamper_resistant_v8.py's recompute block:
resample n_direction harmful/benign prompts, jitter the read layer over {DL-4, DL, DL+4}.
Then compares the spread of that family against the angle to d_surgical.
"""
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import (capture_residuals, empirical_refusal_directions,  # noqa: E402
                         load_model)
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts  # noqa: E402

CKPT = ROOT / "outputs/adapters/tamper_resistant_qwen3_0p6b_v8.pt"
DL, N_DIR, N_CAP, RANK = 20, 256, 256, 16
LAYERS = [DL - 4, DL, DL + 4]
SEEDS = [0, 1, 2, 3]


def load_trained(model, ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    params = dict(model.named_parameters())
    for k, v in sd.items():
        if k in params and params[k].shape == v.shape:
            params[k].data.copy_(v.to(params[k].dtype))


def ang(a, b):
    c = float(torch.dot(a / a.norm(), b / b.norm()).clamp(-1, 1))
    return math.degrees(math.acos(abs(c)))  # sign of d is arbitrary


model, tok, device = load_model("Qwen/Qwen3-0.6B")
load_trained(model, CKPT)
model.eval()

harmful = load_advbench_prompts(None, n=520, seed=42, source="walledai")
benign = list(BENIGN_PROMPTS)

# --- the training ensemble: one entry per (seed, layer), exactly as the recompute block ---
fam = {}
for s in SEEDS:
    rng = random.Random(s)
    hs = rng.sample(harmful, min(N_DIR, len(harmful)))
    bs = rng.sample(benign, min(N_DIR, len(benign)))
    with torch.no_grad():
        dbl = empirical_refusal_directions(model, tok, hs, bs, LAYERS, device)
    for L in LAYERS:
        fam[(s, L)] = dbl[L].float().to(device)
    print(f"  seed {s} done", flush=True)

# --- d_surgical at the canonical layer, same construction as v11_surgical_ablation ---
d20 = fam[(0, DL)]
try:
    from datasets import load_dataset
    gsm = [r["question"] for r in list(load_dataset("openai/gsm8k", "main", split="train"))[: N_CAP // 2]]
except Exception:
    gsm = []
cap_prompts = (gsm + list(BENIGN_PROMPTS))[:N_CAP]
with torch.no_grad():
    H = capture_residuals(model, tok, cap_prompts, [DL], device)[DL].float()
H = H - H.mean(0, keepdim=True)
_, _, Vh = torch.linalg.svd(H, full_matrices=False)
V = Vh[:RANK].to(device)
d_cap = V.T @ (V @ d20)
d_surg = (d20 - d_cap)
d_surg = d_surg / d_surg.norm()

keys = sorted(fam)
same_layer = [ang(fam[(a, L)], fam[(b, L)])
              for L in LAYERS for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]]
cross_layer = [ang(fam[(s, La)], fam[(s, Lb)])
               for s in SEEDS for i, La in enumerate(LAYERS) for Lb in LAYERS[i + 1:]]
all_pairs = [ang(fam[keys[i]], fam[keys[j]])
             for i in range(len(keys)) for j in range(i + 1, len(keys))]
to_surg = [ang(fam[k], d_surg) for k in keys]


def stat(name, xs):
    xs = sorted(xs)
    print(f"{name:<38} n={len(xs):<4} min={xs[0]:6.2f}  med={xs[len(xs)//2]:6.2f}  max={xs[-1]:6.2f}")


print()
stat("resample only (same layer)", same_layer)
stat("layer jitter only (same seed)", cross_layer)
stat("whole training ensemble", all_pairs)
stat("ensemble -> d_surgical", to_surg)
