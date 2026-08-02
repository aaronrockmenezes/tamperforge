"""Do the panel's in-memory attack and v11's export attack produce the SAME weights?"""
import sys, torch
from pathlib import Path
sys.path.insert(0,"src"); sys.path.insert(0,"experiments")
from tamperforge import load_model
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts
from v11_surgical_ablation import _cap_prompts, _cap_subspace, _load_trained, _attack_
from tamperforge import empirical_refusal_direction
from version_a_attack import DirectionBank, SCOPES
from version_a_eval_panel import fixed_panel, attacked_weights
from train_tamper_resistant_v8 import _ablated_overrides

CK="outputs/adapters/tamper_resistant_qwen3_0p6b_v8.pt"; DL=20; K=16
harmful = load_advbench_prompts(None, n=256, seed=42, source="walledai")
benign  = BENIGN_PROMPTS[:256]

# ---- path A: v11 export construction ----
mA,tok,dev = load_model("Qwen/Qwen3-0.6B"); _load_trained(mA, CK); mA.eval()
with torch.no_grad():
    dA = empirical_refusal_direction(mA, tok, harmful, benign, DL, dev).to(dev)
    V  = _cap_subspace(mA, tok, dev, _cap_prompts(256), DL, K)
    dcap = V.T @ (V @ dA); dsA = (dA - dcap); dsA = dsA/dsA.norm()
    print(f"[A] overlap {float(dcap.norm()/dA.norm()):.4f}")
    _attack_(mA, dsA, list(range(len(mA.model.layers))))

# ---- path B: panel construction ----
mB,_,_ = load_model("Qwen/Qwen3-0.6B"); _load_trained(mB, CK); mB.eval()
with torch.no_grad():
    bank = DirectionBank.build(mB, tok, dev, harmful, benign, _cap_prompts(256),
                               layers=list(range(len(mB.model.layers))),
                               read_layers=(DL,), cap_ranks=(4,K))
    print(f"[B] overlap {bank.overlap[(DL,K)]:.4f}")
    spec = [s for s in fixed_panel(len(mB.model.layers), list(range(10,28)), DL)
            if s.tag=="surgical_k16"][0]
    dB = bank.directions_for(spec)
    print(f"[B] |cos(dsA,dB)| = {abs(float(torch.dot(dsA/dsA.norm(), dB/dB.norm()))):.6f}")
    ov = _ablated_overrides(mB, dB, spec.layers, spec.read_proj, spec.write_proj, spec.alphas)
    print(f"[B] overrides: {len(ov)} matrices")
    pB = dict(mB.named_parameters())
    for k,v in ov.items(): pB[k].copy_(v.to(pB[k].dtype))

# ---- compare ----
pA, pB = dict(mA.named_parameters()), dict(mB.named_parameters())
worst, nbad = 0.0, 0
for k in pA:
    d = (pA[k].float()-pB[k].float()).abs().max().item()
    if d > 1e-3: nbad += 1
    worst = max(worst, d)
print(f"\nmax |A-B| over ALL params: {worst:.6f}   params differing >1e-3: {nbad}/{len(pA)}")
