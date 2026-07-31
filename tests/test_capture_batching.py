"""Batched capture_residuals must equal the batch-1 path, and be much faster.

capture_residuals feeds every refusal direction and capability subspace in the project.
If batching perturbs it, every downstream number moves and nothing else would catch it --
so this compares against batch_size=1 on the real model rather than trusting the argument
that right padding is safe under causal attention.

Needs a GPU and the Qwen weights; run it on the box:
    python tests/test_capture_batching.py
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import capture_residuals, load_model  # noqa: E402
from tamperforge.data import BENIGN_PROMPTS  # noqa: E402

LAYERS = [10, 16, 20, 24]


def main() -> None:
    model, tok, device = load_model("Qwen/Qwen3-0.6B")
    model.eval()
    prompts = list(BENIGN_PROMPTS)[:64]
    print(f"device={device} prompts={len(prompts)} layers={LAYERS}")

    t0 = time.time()
    with torch.no_grad():
        ref = capture_residuals(model, tok, prompts, LAYERS, device, batch_size=1)
    t_seq = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        got = capture_residuals(model, tok, prompts, LAYERS, device, batch_size=32)
    t_batch = time.time() - t0

    # CONTROL: two batched runs at different widths. bf16 matmuls reduce in a different
    # order per kernel/tile shape, so this is the noise floor batching cannot go below.
    # Any 1-vs-32 gap of the same size is that roundoff, not a masking/indexing bug.
    with torch.no_grad():
        ctrl = capture_residuals(model, tok, prompts, LAYERS, device, batch_size=16)

    print("\nper-activation (cosine):")
    for li in LAYERS:
        assert ref[li].shape == got[li].shape, (li, ref[li].shape, got[li].shape)
        c1 = torch.nn.functional.cosine_similarity(ref[li], got[li], dim=-1).min().item()
        c2 = torch.nn.functional.cosine_similarity(ctrl[li], got[li], dim=-1).min().item()
        print(f"  layer {li:>2}: 1-vs-32 min_cos={c1:.6f}   16-vs-32 min_cos={c2:.6f} (noise floor)")
        assert c1 >= c2 - 1e-4, (
            f"layer {li}: 1-vs-32 ({c1:.6f}) is worse than the batch-width noise floor "
            f"({c2:.6f}) -- that is a real bug, not roundoff")

    # WHAT ACTUALLY MATTERS: the real refusal direction. This must be harmful-vs-benign,
    # not an arbitrary split of one pool -- a benign-vs-benign "direction" has no signal,
    # so its cosine is noise/noise and says nothing about whether batching is safe.
    from tamperforge.data import load_advbench_prompts
    harmful = load_advbench_prompts(ROOT / "data" / "advbench_harmful_behaviors.csv",
                                    n=128, seed=42, source="local")
    benign = (list(BENIGN_PROMPTS) * 8)[:128]
    print(f"\nreal refusal direction (harmful={len(harmful)} vs benign={len(benign)}):")
    with torch.no_grad():
        h1 = capture_residuals(model, tok, harmful, LAYERS, device, batch_size=1)
        b1 = capture_residuals(model, tok, benign, LAYERS, device, batch_size=1)
        h32 = capture_residuals(model, tok, harmful, LAYERS, device, batch_size=32)
        b32 = capture_residuals(model, tok, benign, LAYERS, device, batch_size=32)
    for li in LAYERS:
        def _dir(h, b):
            d = h[li].mean(0) - b[li].mean(0)
            return d / d.norm()
        c = torch.dot(_dir(h1, b1), _dir(h32, b32)).item()
        print(f"  layer {li:>2}: cos(d_batch1, d_batch32) = {c:.8f}")
        assert c > 0.999, f"layer {li} direction diverged: cos={c}"

    print(f"\nbatch1={t_seq:.1f}s  batch32={t_batch:.1f}s  speedup={t_seq / max(t_batch, 1e-9):.1f}x")
    assert t_batch < t_seq, "batching was not faster"
    print("ok")


if __name__ == "__main__":
    main()
