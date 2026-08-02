"""Validate the in-loop ARC/MMLU probes against a checkpoint with KNOWN answers.

Gate from TODO.md: these have never been run. v8 clean scores ARC acc 0.3148 (local) /
0.3217 (box) and MMLU 0.437 across the campaign's 12 subjects. If the probes do not land
near those, no in-loop reading from them can be trusted.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import torch
from tamperforge import load_model
from train_tamper_resistant_v8 import _arc_rows, _mmlu_rows, _clean_mc_probe
from v11_surgical_ablation import _load_trained

m, tok, dev = load_model("Qwen/Qwen3-0.6B")
_load_trained(m, "outputs/adapters/tamper_resistant_qwen3_0p6b_v8.pt")
m.eval()
for name, loader, n, known in [("ARC", _arc_rows, 300, "0.3148 local / 0.3217 box"),
                               ("MMLU", _mmlu_rows, 300, "0.437 (12 subjects)")]:
    t0 = time.time()
    rows = loader(n)
    acc = _clean_mc_probe(m, tok, dev, rows)
    print(f"{name:<5} n={len(rows):<4} acc={acc:.4f}   known: {known}   ({time.time()-t0:.1f}s)")
