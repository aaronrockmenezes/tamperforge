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
def _scores(clean, atk, tok, device, prompt: str, n_new: int):
    enc = apply_chat_template_no_think(tok, [{"role": "user", "content": prompt}],
                                       return_tensors="pt", return_dict=True,
                                       add_generation_prompt=True).to(device)
    plen = enc["input_ids"].shape[1]
    clean.config.use_cache = True
    full = clean.generate(**enc, max_new_tokens=n_new, do_sample=False,
                          use_cache=True, pad_token_id=tok.eos_token_id)
    if full.shape[1] <= plen:
        return None
    la = atk(input_ids=full).logits.float()[:, :-1, :]
    lc = clean(input_ids=full).logits.float()[:, :-1, :]
    tgt = full[:, 1:]
    m = torch.zeros_like(tgt, dtype=torch.bool)
    m[:, plen - 1:] = True                       # continuation tokens only
    if not m.any():
        return None
    ce = torch.nn.functional.cross_entropy(
        la.reshape(-1, la.size(-1)), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
    gib = float(torch.clamp(ce[m], max=30.0).mean())   # fp32 + clamp, as in training
    kl = float(torch.nn.functional.kl_div(
        torch.log_softmax(la, -1), torch.log_softmax(lc, -1),
        reduction="none", log_target=True).sum(-1)[m].clamp(max=30.0).mean())
    return gib, kl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--attacked", required=True)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--n-new", type=int, default=32, help="matches --gib-gen-tokens default")
    ap.add_argument("--limit", type=int, default=-1)
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
    for i, p in enumerate(prompts):
        s = _scores(clean, atk, tok, device, p, args.n_new)
        if s is None:
            continue
        d = data.setdefault(labels[p], {"gib_ce": [], "kl": []})
        d["gib_ce"].append(s[0])
        d["kl"].append(s[1])
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(prompts)}")

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
