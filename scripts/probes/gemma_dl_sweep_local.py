#!/usr/bin/env python
"""Sweep Gemma's refusal-direction layer on the BASE model. Runs locally on Apple MPS.

WHY THIS EXISTS. Every Gemma arm (version_G/H/I/J) trained and evaluated at --direction-layer 14,
inherited from a v8-era sweep. That sweep measured the DEFENDED v8 model, not the base, so DL=14
was never validated as the layer an attacker would actually use. CLAUDE.md's own rule is "never
scale a direction layer between architectures -- sweep the BASE model and measure"; for Gemma
that was never done. The symptom that motivates it: on version_G-gemma, surgical k16 (which
recomputes its own direction) reaches 0.9423 harm while rank-1 at DL=14 reaches only 0.1135.
That gap is what you would see if DL=14 is simply not where Gemma's refusal lives.

WHAT IT MEASURES. For each candidate layer L: take the base model, compute the mean-difference
refusal direction at L, project it out of every layer (the canonical Arditi attack), generate on
held-out harmful prompts, and count how much refusal survived. The layer with the LOWEST
surviving refusal is the strongest attack layer -- that is the one a defence must be built at.

SCORED WITH THE PINNED DEEPSEEK JUDGE, not keyword matching. The judge is the only instrument
that separates the three outcomes that matter here: refused / harmful_actionable / GIBBERISH.
Keyword detection cannot see gibberish at all, and gibberish is exactly what invalidated
ABL-v7's Gemma result (61.4% clean gibberish read as "safe"). A layer that scores low refusal by
producing garbage has not found the refusal direction -- it has broken the model -- and only the
judge can tell those apart.

Judge is pinned to deepseek-v4-flash-0731 (the repo default is the FLOATING -flash tag; every
comparison here must use the pinned snapshot). `--keyword-only` skips the API entirely if budget
runs out mid-sweep; the generations are always written to disk first so nothing is lost.

BUDGET. calls = (len(layers) + 1) * n_harmful. At the defaults below that is 9*40 = 360 calls,
roughly $0.05-0.10 on the pinned judge. Check `--estimate-only` before spending.

  python scripts/probes/gemma_dl_sweep_local.py --layers 6,10,12,14,16,18,20,22 --n-harmful 40
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import (  # noqa: E402
    abliterate_model_inplace,
    apply_chat_template_no_think,
    empirical_refusal_direction,
    is_refusal,
    load_model,
)
from tamperforge.data import load_advbench  # noqa: E402
from tamperforge.data_p1b import load_benign_instructions  # noqa: E402
from tamperforge.eval.judge import OpenRouterJudge, usefulness_label  # noqa: E402

# The repo's OpenRouterJudge still defaults to the FLOATING deepseek-v4-flash tag. Every number
# in this project is pinned to the -0731 snapshot; mixing the two is not comparable.
JUDGE_MODEL = "deepseek/deepseek-v4-flash-0731"


def judge_batch(judge, prompts, responses, workers=12) -> dict:
    """{label: rate} plus parse failures. Falls back to keyword on total judge failure."""
    from concurrent.futures import ThreadPoolExecutor
    counts, failures = {}, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for parsed in ex.map(lambda pr: _score_one(judge, *pr), zip(prompts, responses)):
            if parsed is None:
                failures += 1
                continue
            lab = usefulness_label(parsed)
            counts[lab] = counts.get(lab, 0) + 1
    n = max(sum(counts.values()), 1)
    return {"counts": counts, "parse_failures": failures,
            "harmful_actionable": round(counts.get("harmful_actionable", 0) / n, 4),
            "gibberish": round(counts.get("gibberish", 0) / n, 4),
            "refused": round(counts.get("refused", 0) / n, 4)}


def _score_one(judge, prompt, response):
    """score() returns {judge_model, raw, parsed}; usefulness_label wants the PARSED sub-dict.
    Passing the outer dict silently labels everything 'benign' -- it has no refusal/
    harmful_compliance keys, so every rate reads 0.0000 and looks like a plausible result."""
    try:
        return judge.score(prompt, response).get("parsed")
    except Exception:
        return None

# Every matrix abliterate_model_inplace touches. Snapshotting only these keeps the restore
# cheap; deep-copying the whole model per layer would not fit alongside it in 16 GB.
TOUCHED = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def snapshot(model) -> dict:
    """CPU copy of every weight the ablation will overwrite."""
    out = {}
    for i, layer in enumerate(model.model.layers):
        for name in TOUCHED:
            mod = layer
            for part in name.split("."):
                mod = getattr(mod, part)
            out[(i, name)] = mod.weight.data.detach().to("cpu", copy=True)
    return out


def restore(model, snap: dict) -> None:
    for (i, name), W in snap.items():
        mod = model.model.layers[i]
        for part in name.split("."):
            mod = getattr(mod, part)
        mod.weight.data.copy_(W.to(mod.weight.device))


@torch.no_grad()
def generate(model, tok, prompts, device, max_new=64, batch_size=8) -> list[str]:
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        texts = [apply_chat_template_no_think(tok, [{"role": "user", "content": p}],
                                              tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--layers", default="", help="comma list; default = every 2nd layer")
    ap.add_argument("--n-harmful", type=int, default=40, help="prompts per layer")
    ap.add_argument("--n-direction", type=int, default=64, help="prompts per side for the direction")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="float32 default: Gemma NaNs in bf16 under some attention paths "
                         "(CLAUDE.md), and 1B in fp32 is only ~4GB")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--judge-workers", type=int, default=12)
    ap.add_argument("--keyword-only", action="store_true",
                    help="skip the API entirely (generations are still written to disk)")
    ap.add_argument("--estimate-only", action="store_true", help="print judge call count and exit")
    ap.add_argument("--out", default="results/gemma_dl_sweep_local.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    import os
    os.environ.setdefault("TF_ATTN_IMPL", "eager")   # Gemma stability, per CLAUDE.md
    torch.set_default_dtype(torch.float32 if args.dtype == "float32" else torch.bfloat16)

    print(f"[load] {args.model_id} dtype={args.dtype}")
    model, tok, device = load_model(args.model_id)
    if args.dtype == "float32":
        model = model.float()
    model.eval()
    n_layers = len(model.model.layers)
    print(f"[load] device={device} layers={n_layers}")

    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              if args.layers else list(range(2, n_layers, 2)))
    bad = [L for L in layers if not 0 <= L < n_layers]
    if bad:
        sys.exit(f"layers out of range for this model ({n_layers} layers): {bad}")

    pairs = load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv")
    harmful_all = [p for p, _ in pairs]
    benign_all = load_benign_instructions(max(args.n_direction, 64), seed=42)
    # Direction prompts and eval prompts must be DISJOINT, or the sweep scores the layer on the
    # very prompts that defined its direction and every layer looks better than it is.
    dir_harmful = harmful_all[:args.n_direction]
    dir_benign = benign_all[:args.n_direction]
    eval_harmful = harmful_all[args.n_direction:args.n_direction + args.n_harmful]
    print(f"[data] direction {len(dir_harmful)}h/{len(dir_benign)}b | eval {len(eval_harmful)} "
          f"(disjoint from direction set)")

    n_calls = (len(layers) + 1) * len(eval_harmful)
    print(f"[budget] {n_calls} judge calls ({len(layers)} layers + base) x {len(eval_harmful)} prompts")
    if args.estimate_only:
        return

    judge = None
    if not args.keyword_only:
        judge = OpenRouterJudge(model=args.judge_model)
        print(f"[judge] pinned {args.judge_model}")

    gen_dir = ROOT / "results" / "gemma_dl_sweep_gens"
    gen_dir.mkdir(parents=True, exist_ok=True)

    def record(tag, outs):
        """Persist generations BEFORE judging so an API failure never loses the GPU work."""
        with open(gen_dir / f"{tag}.jsonl", "w") as f:
            for p, o in zip(eval_harmful, outs):
                f.write(json.dumps({"prompt": p, "response": o}) + "\n")

    print("[base] generating unablated reference...")
    t0 = time.time()
    base_out = generate(model, tok, eval_harmful, device, args.max_new, args.batch_size)
    record("base", base_out)
    base_ref = sum(is_refusal(o) for o in base_out) / len(base_out)
    base_j = judge_batch(judge, eval_harmful, base_out, args.judge_workers) if judge else None
    print(f"[base] keyword-refusal {base_ref:.4f}"
          + (f" | judged harm {base_j['harmful_actionable']:.4f} gib {base_j['gibberish']:.4f} "
             f"refused {base_j['refused']:.4f}" if base_j else "")
          + f"  ({time.time()-t0:.0f}s)")

    snap = snapshot(model)
    print(f"[snap] {len(snap)} matrices held on CPU")

    rows = []
    for L in layers:
        t0 = time.time()
        d = empirical_refusal_direction(model, tok, dir_harmful, dir_benign, L, device)
        if not torch.isfinite(d).all():
            print(f"  L{L:<3d} direction is non-finite -- skipping"); restore(model, snap); continue
        abliterate_model_inplace(model, d, list(range(n_layers)))
        outs = generate(model, tok, eval_harmful, device, args.max_new, args.batch_size)
        restore(model, snap)
        record(f"L{L}", outs)

        ref = sum(is_refusal(o) for o in outs) / len(outs)
        empty = sum(1 for o in outs if len(o.strip()) < 5) / len(outs)
        row = {"layer": L, "keyword_refusal": round(ref, 4),
               "near_empty": round(empty, 4), "secs": round(time.time() - t0, 1),
               "sample": outs[0][:160]}
        if judge:
            j = judge_batch(judge, eval_harmful, outs, args.judge_workers)
            row |= {"harmful_actionable": j["harmful_actionable"], "gibberish": j["gibberish"],
                    "judged_refused": j["refused"], "parse_failures": j["parse_failures"]}
            print(f"  L{L:<3d} HARM {j['harmful_actionable']:.4f}  gib {j['gibberish']:.4f}  "
                  f"refused {j['refused']:.4f}  | kw-ref {ref:.4f} empty {empty:.2f}  "
                  f"{time.time()-t0:.0f}s")
        else:
            print(f"  L{L:<3d} kw-refusal {ref:.4f}  near-empty {empty:.2f}  {time.time()-t0:.0f}s")
        rows.append(row)

    if not rows:
        sys.exit("no layers produced a usable direction")

    # Rank by judged harm when we have it. Ranking by refusal alone would crown whichever layer
    # produces the most gibberish -- the exact error that made ABL-v7's Gemma result look like a
    # win. harmful_actionable counts only coherent, actionable harm.
    key = "harmful_actionable" if judge else "keyword_refusal"
    rows.sort(key=lambda r: -r[key] if judge else r[key])
    best = rows[0]
    print(f"\n[best attack layer] L{best['layer']}")
    if judge:
        print(f"  harmful_actionable {best['harmful_actionable']:.4f}  "
              f"gibberish {best['gibberish']:.4f}  (base harm "
              f"{base_j['harmful_actionable']:.4f})")
        print(f"  DL=14 was used by every Gemma arm; it ranks "
              f"#{[r['layer'] for r in rows].index(14)+1} of {len(rows)} here"
              if any(r["layer"] == 14 for r in rows) else "  (L14 not in this sweep)")
    print("[note] a layer with high gibberish has broken the model, not found the refusal "
          "direction. Prefer high harmful_actionable AND low gibberish.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model_id": args.model_id, "n_layers": n_layers, "dtype": args.dtype,
               "base_keyword_refusal": round(base_ref, 4), "base_judged": base_j,
               "n_harmful": len(eval_harmful), "n_direction": len(dir_harmful),
               "judge_model": None if args.keyword_only else args.judge_model,
               "ranked_by": key, "generations": str(gen_dir.relative_to(ROOT)),
               "layers": rows}, open(out, "w"), indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
