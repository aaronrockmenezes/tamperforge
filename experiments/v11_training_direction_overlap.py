"""Capability overlap of the directions v8 ACTUALLY trains against.

Hypothesis under test: the training ensemble never shows the model a low-capability-overlap
ablation, so the model learns "collapse when a HIGH-overlap direction is removed" -- which
d_surgical (~0 overlap by construction) simply does not trigger.

Reproduces the recompute block in train_tamper_resistant_v8.py exactly: resample
n_direction harmful/benign prompts per refresh, jitter the read layer over {DL-4, DL, DL+4},
mean-diff estimator, one shared direction applied to every attacked layer.

Reports each training direction's overlap with the capability subspace both in its own
layer's frame and in the layer-20 frame the surgical attack was built in, at cap-rank 4
(k_min, where the escape hatch opens) and cap-rank 16 (what the known break used).
"""
import json
import random
import sys
from pathlib import Path

import torch

TF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TF / "src"))
sys.path.insert(0, str(TF / "experiments"))

from tamperforge import (capture_residuals, empirical_refusal_directions,  # noqa: E402
                         load_model)
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts  # noqa: E402
from v11_surgical_ablation import _cap_prompts, _load_trained  # noqa: E402

DL, N = 20, 256
LAYERS = [DL - 4, DL, DL + 4]
SEEDS = [0, 1, 2, 3]
RANKS = [4, 16]
CKPT = TF / "outputs/adapters/tamper_resistant_qwen3_0p6b_v8.pt"

model, tok, device = load_model("Qwen/Qwen3-0.6B")
_load_trained(model, str(CKPT))
model.eval()

harmful_pool = load_advbench_prompts(None, n=520, seed=42, source="walledai")
benign_pool = list(BENIGN_PROMPTS)

# capability subspaces at each read layer, one capture pass for all layers
with torch.no_grad():
    Hc = capture_residuals(model, tok, _cap_prompts(N), LAYERS, device)
cap = {}
for L in LAYERS:
    H = Hc[L].float()
    H = H - H.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(H, full_matrices=False)
    cap[L] = {r: Vh[:r].to(device) for r in RANKS}

# the training directions
dirs = {}
for s in SEEDS:
    rng = random.Random(s)
    hs = rng.sample(harmful_pool, min(N, len(harmful_pool)))
    bs = rng.sample(benign_pool, min(N, len(benign_pool)))
    with torch.no_grad():
        dbl = empirical_refusal_directions(model, tok, hs, bs, LAYERS, device)
    for L in LAYERS:
        d = dbl[L].float().to(device)
        dirs[(s, L)] = d / d.norm()
    print(f"  seed {s} done", flush=True)


def overlap(d, V):
    return float(((V.T @ (V @ d)).norm() / d.norm()).clamp(0, 1))


rows, report = [], {}
for r in RANKS:
    own, at20 = [], []
    for (s, L), d in sorted(dirs.items()):
        o_own, o_20 = overlap(d, cap[L][r]), overlap(d, cap[DL][r])
        own.append(o_own)
        at20.append(o_20)
        rows.append((r, s, L, o_own, o_20))
    report[f"rank{r}"] = {
        "own_frame": {"min": round(min(own), 4), "med": round(sorted(own)[len(own) // 2], 4),
                      "max": round(max(own), 4)},
        "layer20_frame": {"min": round(min(at20), 4), "med": round(sorted(at20)[len(at20) // 2], 4),
                          "max": round(max(at20), 4)},
    }

print("\ncap_rank  seed  layer   overlap(own L)   overlap(L20 frame)")
for r, s, L, a, b in rows:
    print(f"{r:>8}  {s:>4}  {L:>5}   {a:>13.4f}   {b:>17.4f}")
print("\nsummary:", json.dumps(report, indent=2))
print("\nd_surgical overlap with its own cap subspace = 0.0 by construction.")
(TF / "results/v11_surgical_overlap/training_direction_overlap.json").write_text(
    json.dumps({"per_direction": [{"cap_rank": r, "seed": s, "layer": L,
                                   "overlap_own_frame": round(a, 4),
                                   "overlap_layer20_frame": round(b, 4)}
                                  for r, s, L, a, b in rows], "summary": report}, indent=2))
