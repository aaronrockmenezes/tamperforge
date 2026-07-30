"""Diagnostic: does gib_ce (or KL) separate normal / harmful / gibberish responses?

The question this settles: gib_ce's job is to tell "the attack broke the model" from "the
attack left a working model". If it cannot distinguish the three outcomes an attacked model
actually produces, then no value of --gap-target fixes it, because a threshold only works if
the categories are ORDERED consistently.

Reproduces `_argmax_divergence_loss` (train_tamper_resistant_v8.py) exactly:
  1. CLEAN model greedily generates n_new tokens for the prompt;
  2. the ATTACKED model is teacher-forced on (prompt + that continuation);
  3. gib_ce = mean CE on the continuation tokens, fp32, clamped at 30.
The one difference is plumbing: training applies the attack as functional_call overrides on
one model, this loads a separately materialised attacked checkpoint. Same arithmetic.

Also computes KL(clean || attacked) at the same positions, because "would KL work better"
is worth measuring rather than asserting. Prediction on record: no. gib_ce is
divergence-from-what-the-clean-model-would-say, and KL is the same comparison with the clean
model's entropy subtracted off -- coherent harm diverges from a clean continuation exactly
as much as gibberish does. Changing the divergence functional does not touch that.

Rows are grouped by the label the ATTACKED model earned on that prompt (from the judged
run), so this is a within-checkpoint measurement -- no cross-model confound.

Best target: the surgical k16 arm, which has all three categories in one checkpoint
(233 harmful / 191 refused / 84 gibberish).

  python experiments/v11_gibce_separation.py \
    --clean outputs/v11_qwen_v8_clean \
    --attacked outputs/v11_v8_surg_k16 \
    --judged results/v11_v11_v8_surg_k16_judged
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import apply_chat_template_no_think, load_model  # noqa: E402
from tamperforge.eval.judge import usefulness_label  # noqa: E402

# usefulness_label -> the three buckets asked about. harmful_vague is dropped: it is
# neither cleanly harmful nor cleanly normal and would blur every pair.
BUCKET = {"refused": "normal", "benign": "normal",
          "harmful_actionable": "harmful", "gibberish": "gibberish"}


def _auc(a: list[float], b: list[float]) -> float:
    """P(x drawn from a > y drawn from b). 0.5 = indistinguishable."""
    if not a or not b:
        return float("nan")
    s = torch.tensor(a + b)
    y = torch.tensor([1] * len(a) + [0] * len(b))
    r = s.argsort().argsort().float()
    return ((r[y == 1].sum() - len(a) * (len(a) - 1) / 2) / (len(a) * len(b))).item()


def _best_threshold_bacc(a: list[float], b: list[float]) -> float:
    """Best balanced accuracy any single threshold can achieve on this pair.

    This is the direct answer to 'would a larger --gap-target fix it?'. A threshold is
    exactly what gap_target is. ~0.5 means no threshold anywhere separates the two.
    """
    if not a or not b:
        return float("nan")
    best = 0.0
    for t in sorted(set(a + b)):
        for sign in (1, -1):
            hit_a = sum(1 for x in a if sign * x >= sign * t) / len(a)
            hit_b = sum(1 for x in b if sign * x < sign * t) / len(b)
            best = max(best, (hit_a + hit_b) / 2)
    return best


def _labels(judged: Path) -> dict[str, str]:
    out = {}
    for line in open(judged / "judgments.jsonl"):
        r = json.loads(line)
        j = r.get("judge")
        if isinstance(j, str):
            j = ast.literal_eval(j)
        p = (j or {}).get("parsed") or {}
        if p.get("refusal") is None:
            continue  # judge parse failure -- never silently bucket it
        b = BUCKET.get(usefulness_label(p))
        if b:
            out[r["prompt"]] = b
    return out


@torch.no_grad()
def _scores_batch(clean, atk, tok, device, prompts: list[str], n_new: int,
                  stop_at_eos: bool = True):
    """Per-prompt (gib_ce, kl) for a batch. Returns list aligned with `prompts`.

    Left-padded so every prompt ends at the same position -- the continuation then starts
    at the same index for the whole batch, which keeps the masking simple. Logits are
    sliced to the continuation window BEFORE the fp32 cast: the full [B, L, 152k] tensor in
    fp32 is ~3GB at batch 32, the 32-token window is ~0.6GB.
    """
    texts = [apply_chat_template_no_think(tok, [{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True)
             for p in prompts]
    old_side, tok.padding_side = tok.padding_side, "left"
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    tok.padding_side = old_side
    plen = enc["input_ids"].shape[1]

    clean.config.use_cache = True
    full = clean.generate(**enc, max_new_tokens=n_new, do_sample=False,
                          use_cache=True, pad_token_id=tok.eos_token_id)
    if full.shape[1] <= plen:
        return [None] * len(prompts)
    gen = full[:, plen:]                                  # [B, n_gen]
    n_gen = gen.shape[1]
    am = torch.cat([enc["attention_mask"],
                    torch.ones_like(gen)], dim=1)

    # FIDELITY: training calls _argmax_divergence_loss one prompt at a time, so generate
    # returns each sequence truncated at its own EOS and never pads. Batching pads short
    # generations out to the longest in the batch, and scoring that padding is a pure
    # batching artifact -- it shifted the measured means by 0.3-2.0 nats in testing.
    # Stopping at EOS is therefore what reproduces the training signal, not an improvement
    # on it; verified against a batch-1 run (normal 2.0509 vs 2.0503, harmful 2.2568 vs
    # 2.2654). Residual per-sequence differences come from greedy decoding under
    # left-padding occasionally flipping an argmax; noise, not bias.
    if stop_at_eos:
        is_eos = gen == tok.eos_token_id
        first_eos = torch.where(is_eos.any(1), is_eos.float().argmax(1),
                                torch.full((gen.shape[0],), n_gen, device=gen.device).float())
        valid = torch.arange(n_gen, device=gen.device)[None, :] <= first_eos[:, None]
    else:
        valid = torch.ones_like(gen, dtype=torch.bool)

    # Left-padding means a bare forward would assign RoPE positions by plain arange, so
    # every padded sequence gets shifted positions and wrong logits. generate() derives
    # position_ids from the mask internally; an explicit forward does not, so do it here.
    # Without this the batched numbers do not match the batch-1 reference.
    pos = (am.cumsum(-1) - 1).clamp(min=0)

    # logits at positions plen-1 .. end-2 predict exactly the continuation tokens
    la = atk(input_ids=full, attention_mask=am, position_ids=pos).logits[:, plen - 1:-1, :].float()
    lc = clean(input_ids=full, attention_mask=am, position_ids=pos).logits[:, plen - 1:-1, :].float()
    ce = torch.nn.functional.cross_entropy(
        la.reshape(-1, la.size(-1)), gen.reshape(-1), reduction="none").reshape(gen.shape)
    ce = torch.clamp(ce, max=30.0)                        # fp32 + clamp, as in training
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(la, -1), torch.log_softmax(lc, -1),
        reduction="none", log_target=True).sum(-1).clamp(max=30.0)

    n = valid.sum(1)
    out = []
    for i in range(gen.shape[0]):
        if n[i] < 1:
            out.append(None)
            continue
        out.append((float(ce[i][valid[i]].mean()), float(kl[i][valid[i]].mean())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--attacked", required=True)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--n-new", type=int, default=32, help="matches --gib-gen-tokens default")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--score-eos-padding", action="store_true",
                    help="also score the eos-padding batching appends after an early stop. "
                         "OFF by default: training calls _argmax_divergence_loss one prompt "
                         "at a time, so generate returns each sequence truncated at its own "
                         "EOS and never pads. Scoring the padding is a batching artifact, "
                         "and leaving it on shifts means by ~0.3-2.0 nats.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/v11_gibce_separation.json")
    args = ap.parse_args()

    labels = _labels(ROOT / args.judged if not Path(args.judged).is_absolute()
                     else Path(args.judged))
    prompts = list(labels)
    if args.limit > 0:
        prompts = prompts[: args.limit]
    from collections import Counter
    print(f"[sep] {len(prompts)} labelled prompts: {dict(Counter(labels[p] for p in prompts))}")

    clean, tok, device = load_model(args.clean, device=args.device)
    clean.eval()
    atk, _, _ = load_model(args.attacked, device=args.device)
    atk.eval()

    data: dict[str, dict[str, list[float]]] = {}
    for i in range(0, len(prompts), args.batch_size):
        chunk = prompts[i: i + args.batch_size]
        for p, s in zip(chunk, _scores_batch(clean, atk, tok, device, chunk, args.n_new,
                                                  not args.score_eos_padding),
                        strict=True):
            if s is None:
                continue
            d = data.setdefault(labels[p], {"gib_ce": [], "kl": []})
            d["gib_ce"].append(s[0])
            d["kl"].append(s[1])
        print(f"  ...{min(i + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)

    report = {"n": {k: len(v["gib_ce"]) for k, v in data.items()}, "metrics": {}}
    for metric in ("gib_ce", "kl"):
        print(f"\n=== {metric} ===")
        means = {k: sum(v[metric]) / len(v[metric]) for k, v in data.items() if v[metric]}
        for k in ("normal", "harmful", "gibberish"):
            if k in means:
                print(f"  {k:10s} n={len(data[k][metric]):4d}  mean {means[k]:8.4f}")
        m = {"means": means, "pairs": {}}
        for a, b in [("normal", "gibberish"), ("normal", "harmful"), ("harmful", "gibberish")]:
            if a not in data or b not in data:
                continue
            va, vb = data[a][metric], data[b][metric]
            delta = means[a] - means[b]
            auc = _auc(va, vb)
            bacc = _best_threshold_bacc(va, vb)
            m["pairs"][f"{a}_vs_{b}"] = {"delta": delta, "auc": auc, "best_threshold_bacc": bacc}
            print(f"  {a:9s} vs {b:9s}  delta {delta:+8.4f}   AUC {auc:.3f}   "
                  f"best-threshold bal-acc {bacc:.3f}")
        report["metrics"][metric] = m

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[sep] wrote {out}")
    print("READ: 'best-threshold bal-acc' is the direct answer to 'would a larger\n"
          "--gap-target fix this?' -- gap_target IS a threshold. ~0.5 means no threshold\n"
          "anywhere separates that pair, so no target value helps. The pair that matters\n"
          "most is harmful vs gibberish: that is the distinction the defence depends on.")


if __name__ == "__main__":
    main()
