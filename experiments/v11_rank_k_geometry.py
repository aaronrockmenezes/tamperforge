"""Rank-k refusal subspace geometry, base vs v8.

The question: is v8's capability entanglement spread across the refusal subspace, or
concentrated in the top direction? If only direction 1 is entangled, a rank-k attacker
escapes the trap for free by using directions 2..k -- no surgical projection needed.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tamperforge import capture_residuals, empirical_refusal_direction, load_model  # noqa: E402
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts  # noqa: E402
from v11_surgical_ablation import _cap_prompts, _cap_subspace, _load_trained  # noqa: E402

DL, N, CAP_RANK, KMAX = 20, 256, 16, 16
KS = [1, 2, 4, 8, 16]
CKPT = ROOT / "outputs/adapters/tamper_resistant_qwen3_0p6b_v8.pt"

harmful = load_advbench_prompts(None, n=N, seed=42, source="walledai")
harmless = BENIGN_PROMPTS[:N]
report = {}

for tag, ckpt in [("base", None), ("v8", str(CKPT))]:
    model, tok, device = load_model("Qwen/Qwen3-0.6B")
    if ckpt:
        _load_trained(model, ckpt)
    model.eval()

    with torch.no_grad():
        Hh = capture_residuals(model, tok, harmful, [DL], device)[DL].float()
        Hb = capture_residuals(model, tok, harmless, [DL], device)[DL].float()
        D = Hh - Hb.mean(0, keepdim=True)
        _, s, Vh = torch.linalg.svd(D, full_matrices=False)
        Vref = Vh[:KMAX].to(device)                       # [16, d_model]
        Vcap = _cap_subspace(model, tok, device, _cap_prompts(N), DL, CAP_RANK)
        d_mean = empirical_refusal_direction(model, tok, harmful, harmless, DL, device).to(device)

        # per-direction capability overlap: ||P_cap v_i|| for each singular direction
        per_dir = ((Vref @ Vcap.T) @ Vcap).norm(dim=1)     # rows are unit norm
        # cos between the mean-diff direction and each SVD direction
        cos_mean = (Vref @ (d_mean / d_mean.norm())).abs()
        energy = (s[:KMAX] ** 2 / (s ** 2).sum()).cpu()

    r = {"per_dir_cap_overlap": [round(float(x), 4) for x in per_dir],
         "cos_meandiff_vs_svd_dir": [round(float(x), 4) for x in cos_mean],
         "singular_energy_frac": [round(float(x), 4) for x in energy],
         "subspace_overlap_by_k": {}}
    for k in KS:
        Rk = Vref[:k]
        Rcap = (Rk @ Vcap.T) @ Vcap
        r["subspace_overlap_by_k"][k] = round(float(Rcap.norm() / Rk.norm()), 4)
        r.setdefault("harmful_energy_by_k", {})[k] = round(float(energy[:k].sum()), 4)
    report[tag] = r

    print(f"\n===== {tag} =====")
    print("dir  cap_overlap  cos(mean-diff)  sing_energy")
    for i in range(KMAX):
        print(f"{i+1:>3}  {per_dir[i]:>11.4f}  {cos_mean[i]:>14.4f}  {energy[i]:>11.4f}")
    print("  k   subspace_cap_overlap   harmful_energy_captured")
    for k in KS:
        print(f"{k:>3}   {r['subspace_overlap_by_k'][k]:>20.4f}   {r['harmful_energy_by_k'][k]:>23.4f}")

    del model
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None

out = ROOT / "results/v11_surgical_overlap/rank_k_geometry.json"
out.write_text(json.dumps(report, indent=2))
print(f"\nwrote {out}")
