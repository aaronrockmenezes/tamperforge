"""Calibrate --gap-target for --gib-mode task.

gap_target=4.0 was tuned for argmax-divergence CE, a different quantity on a different scale.
If it sits ABOVE what the attack can reach, relu(gap - ce) never satisfies and the objective
applies maximum pressure every step -- the FTR-v6 lobotomy failure mode. If it sits BELOW the
untrained attacked CE, it is satisfied from step 1 and trains nothing.

Measures task CE on version_B s500: clean, and under the canonical rank-1 attack.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path(".")
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")

from tamperforge import empirical_refusal_directions, load_model  # noqa: E402
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts  # noqa: E402
from train_tamper_resistant_v8 import (SCOPES, _ablated_overrides,  # noqa: E402
                                       _load_task_qa, _task_degradation_loss)
from v11_surgical_ablation import _load_trained  # noqa: E402

N = 24
model, tok, dev = load_model("Qwen/Qwen3-0.6B", device="cuda")
_load_trained(model, "outputs/version_b_qwen_500.pt")
qa = _load_task_qa(N, seed=0)

with torch.no_grad():
    clean = _task_degradation_loss(model, tok, qa, dev, None).item()

    nl = len(model.model.layers)
    harmful = load_advbench_prompts(None, n=256, seed=42, source="walledai")
    d = empirical_refusal_directions(model, tok, harmful, BENIGN_PROMPTS[:256], [20], dev)[20]
    rp, wp = SCOPES["all"]
    ov = _ablated_overrides(model, d.to(dev), list(range(nl)), rp, wp, None)
    att = _task_degradation_loss(model, tok, qa, dev, ov).item()

print(f"[calib] n={N} pairs")
print(f"[calib] task CE clean            = {clean:.3f}")
print(f"[calib] task CE under rank-1     = {att:.3f}")
print(f"[calib] separation               = {att - clean:.3f}")
gt = max(3.0, round(att + 1.0, 1))
print(f"[calib] SUGGESTED --gap-target   = {gt}")
print(f"[calib] rule: attacked + 1.0, floored at 3.0 -- above what the attack already")
print(f"[calib]       achieves untrained, so there is real work to do, but reachable.")
if att <= clean + 0.2:
    print("[calib] WARNING: rank-1 barely moves task CE on version_B. If the signal is this "
          "weak the objective may not have the leverage assumed.")
