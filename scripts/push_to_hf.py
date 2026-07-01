#!/usr/bin/env python
"""Push tamperforge artifacts to a PRIVATE Hugging Face model repo.

Run from the repo root on the machine that holds the weights (vast):

    python scripts/push_to_hf.py

Requires HF auth with write scope (`hf auth login`, or HF_TOKEN in env).
Uploads (all PRIVATE):
  - adapters/*.pt                         trained SafetyAdapters
  - abliterated_gemma3_1b_it_all_empirical/   full abliterated Gemma checkpoint
  - README.md                            model card

The abliterated checkpoint is a safety-removed derivative of a gated Gemma model.
Keep the repo private; it inherits the Gemma Terms of Use.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "aaronrockmenezes/tamperforge"
ROOT = Path(__file__).resolve().parents[1]

ADAPTERS = [
    "outputs/safety_adapter_p1_cleanbase.pt",
    "outputs/safety_adapter_p1_ablbase.pt",
    "outputs/tamper_resistant_p1b.pt",  # P1b-A trained MLP weights (v5; prose-PPL objective, negative result)
    "outputs/tamper_resistant_p1b_v6.pt",  # v6: argmax objective, matched-attack PASS
    "outputs/tamper_resistant_p1b_v7.pt",  # v7: ensemble attack, ROBUST across the battery
    "outputs/tamper_resistant_p1b_v6_all_argmax.pt",  # v6 all-scope argmax variant
    "outputs/ft_resistant_p4_v2.pt",  # P4-defense v2: TAR FT-resistance, holds K=1 breaks K>=5
    "outputs/ft_resistant_p4_v3.pt",  # P4-defense v3: strong multi-step inner attack (if present)
]
ABLATED_DIR = "outputs/abliterated_gemma3_1b_it_all_empirical"

MODEL_CARD = """---
license: gemma
tags:
  - tamperforge
  - safety
  - abliteration
  - mechanistic-interpretability
base_model: google/gemma-3-1b-it
extra_gated_prompt: >-
  This repo contains a safety-removed (abliterated) derivative of google/gemma-3-1b-it.
  Research use only. Inherits the Gemma Terms of Use.
---

# tamperforge — artifacts

Research artifacts for **tamperforge**: a pre-release procedure that entangles
safety with capability in open-weight LLMs, so cheap automated abliteration
self-defeats (*smart-and-safe XOR dumb-and-dangerous*). Private / unpublished.

Base model: `google/gemma-3-1b-it` (26 layers, d_model 1152, bf16).

## Contents

### `adapters/`
Trained `SafetyAdapter` blocks (nonlinear residual MLP at layer 13:
`h + W_out(SiLU(W_in(h)))`, d_hidden 256). Load via `tamperforge.load_adapter`.
Applied as a forward hook — NOT merged into weights.

- `safety_adapter_p1_cleanbase.pt` — trained on clean Gemma. W_out↔language
  alignment 0.870, benign KL(base‖adapter) 0.0025.
- `safety_adapter_p1_ablbase.pt` — trained on an all-layer refusal-abliterated
  base (adapter as sole safety). Alignment 0.865, KL 0.0026.

- `tamper_resistant_p1b.pt` — P1b-A adversarial-training checkpoint (subset
  state-dict: 78 MLP matrices, all layers). Trained so ablating the empirical
  refusal direction removes safety and raises prose PPL. NEGATIVE RESULT: under a
  real attack + judge, the uncensored trained model is as harmful as the
  uncensored base (judge-ASR 0.60 vs 0.54) — the 8× prose-PPL gap doesn't reduce
  real harm. Evidence that the prose-PPL objective fails.

### `abliterated_gemma3_1b_it_all_empirical/`
**⚠️ Safety-removed model.** Full HF checkpoint of Gemma-3-1b-it with the
empirical refusal direction projected out of all 26 layers (Arditi-style
abliteration). Judge ASR ~0.76, capability intact (ARC-c ~0.43). This is the
"free uncensoring" attacker baseline the project defends against. Research use
only; do not deploy.

## Status

P0 complete. P1 (adapter MAD crux): PASS on the mechanism (ablating the entangled
adapter costs capability random dirs do not, ~400× PPL at k=32; crossover at k=4
on the product), ceiling = coherent usable uncensor (prose-PPL != generation
coherence). P1b-A (weight-level defense): prose-PPL objective ruled out by v5.
Next = a generation-level objective. See the (private) tamperforge repo.
"""


def main() -> None:
    api = HfApi()
    who = api.whoami()
    print(f"[hf] authenticated as {who.get('name')}")

    api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    print(f"[hf] repo ready (private): {REPO_ID}")

    (ROOT / "README_hf.md").write_text(MODEL_CARD)
    api.upload_file(path_or_fileobj=str(ROOT / "README_hf.md"),
                    path_in_repo="README.md", repo_id=REPO_ID, repo_type="model")
    print("[hf] uploaded README.md")

    for rel in ADAPTERS:
        p = ROOT / rel
        if not p.exists():
            print(f"[hf] SKIP missing adapter: {rel}")
            continue
        api.upload_file(path_or_fileobj=str(p),
                        path_in_repo=f"adapters/{p.name}", repo_id=REPO_ID, repo_type="model")
        print(f"[hf] uploaded adapters/{p.name}")

    abl = ROOT / ABLATED_DIR
    if abl.is_dir():
        api.upload_folder(folder_path=str(abl),
                          path_in_repo="abliterated_gemma3_1b_it_all_empirical",
                          repo_id=REPO_ID, repo_type="model")
        print(f"[hf] uploaded {ABLATED_DIR}/")
    else:
        print(f"[hf] SKIP missing model dir: {ABLATED_DIR}")

    print(f"[hf] done -> https://huggingface.co/{REPO_ID} (private)")


if __name__ == "__main__":
    main()
