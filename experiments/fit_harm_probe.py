"""Option F: fit a linear probe that reads "this model is about to comply" off the
residual stream, for use as a training loss. Run BEFORE wiring it into the trainer.

Why a probe. Every reference-text loss tried so far is blind where it matters:
`gib_ce` scores 4 judge categories within 0.09-0.84 nats; `harm_ce` on the AdvBench
affirmative prefix separates refuse-from-comply by 0.206 nats; and swapping in real
mined harmful completions (option A) made it *worse* — 0.041 nats, sign inverted
(see docs/loss_redesign_options.md). Non-circularity and discriminativeness are in
direct tension for any fixed-reference CE. A probe uses no reference text at all: it
is fit directly to the judge labels, so it is discriminative by construction.

Key design choice: the probe reads **prompt-only** residuals, not (prompt+response).
That is what makes it usable as a loss — during training you forward the attacked
model on a harmful prompt, read layer L, apply the probe, and minimise P(comply).
No generation, no sampling, no reference text, fully differentiable.

This works because compliance is linearly predictable from the prompt-conditioned
state; that is the same premise the refusal-direction literature rests on. The
label comes from what the source model ACTUALLY did on that prompt, per the judge.

All layers are fit at once (`capture_residuals` hooks every layer in one pass, so
28 layers cost what 1 costs) and the best is chosen by held-out AUC.

THE HONEST TEST is not overall AUC — with one dominant label per source model, a
probe can score well by learning *model identity* instead of compliance. So this
also reports **within-model AUC**: inside a single checkpoint, does the probe
separate the prompts it complied on from the ones it refused on? That is the number
that decides whether F is real.

  python experiments/fit_harm_probe.py \
    --pairs outputs/qwen_v8_clean=results/t0v2_qwen_v8_clean_judged/judgments.jsonl \
            outputs/attacked_snapshots/heretic_qwen_v8_kl2222_trial160=results/t0v3_strong_think_judged/judgments.jsonl \
    --out data/harm_probe_qwen.pt
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

from tamperforge import capture_residuals, load_model  # noqa: E402
from tamperforge.eval.judge import usefulness_label  # noqa: E402


def _load_labels(path: Path) -> dict[str, int]:
    """{prompt: 1 if the source model produced coherent harm else 0}."""
    out: dict[str, int] = {}
    for line in open(path):
        row = json.loads(line)
        j = row.get("judge")
        if isinstance(j, str):
            j = ast.literal_eval(j)
        parsed = (j or {}).get("parsed") or {}
        if not parsed or parsed.get("refusal") is None:
            continue  # judge parse failure — never let it fall through as a 0
        out[row["prompt"]] = int(usefulness_label(parsed) == "harmful_actionable")
    return out


def _auc(scores: torch.Tensor, y: torch.Tensor) -> float:
    """Rank AUC. 0.5 = chance. Returns nan if only one class present."""
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = torch.cat([pos, neg]).argsort().argsort().float()
    return ((r[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))).item()


def _fit(X: torch.Tensor, y: torch.Tensor, steps: int = 400, l2: float = 1e-2):
    """Logistic regression, torch only. ponytail: plain full-batch GD, no sklearn dep."""
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    Z = (X - mu) / sd
    w = torch.zeros(Z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, y) + l2 * w.pow(2).sum()
        loss.backward()
        opt.step()
    return w.detach(), b.detach(), mu, sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", required=True, metavar="MODEL_DIR=JUDGMENTS",
                    help="source checkpoint and the judged run it produced")
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="fraction of PROMPTS held out (split by prompt, not by row, so "
                         "the probe cannot memorise a prompt seen under another model)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="data/harm_probe.pt")
    args = ap.parse_args()

    feats: dict[int, list[torch.Tensor]] = {}
    ys: list[int] = []
    srcs: list[str] = []
    prompts_all: list[str] = []

    for spec in args.pairs:
        mdir, jpath = spec.split("=", 1)
        labels = _load_labels(ROOT / jpath if not Path(jpath).is_absolute() else Path(jpath))
        prompts = list(labels)
        print(f"[probe] {mdir}: {len(prompts)} labelled prompts, "
              f"{sum(labels.values())} harmful_actionable")
        model, tok, device = load_model(mdir, device=args.device)
        layers = list(range(len(model.model.layers)))
        with torch.no_grad():
            res = capture_residuals(model, tok, prompts, layers, device)
        for li in layers:
            feats.setdefault(li, []).append(res[li])
        ys += [labels[p] for p in prompts]
        srcs += [mdir] * len(prompts)
        prompts_all += prompts
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    y = torch.tensor(ys, dtype=torch.float32)
    # split by PROMPT so the same prompt never lands in both train and test
    uniq = sorted(set(prompts_all))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(uniq), generator=g)
    test_p = {uniq[i] for i in perm[: int(len(uniq) * args.holdout)].tolist()}
    te = torch.tensor([p in test_p for p in prompts_all])
    tr = ~te
    print(f"[probe] {int(tr.sum())} train / {int(te.sum())} test rows, "
          f"base rate {y.mean():.3f}")

    best = None
    for li in sorted(feats):
        X = torch.cat(feats[li])
        w, b, mu, sd = _fit(X[tr], y[tr])
        s_te = ((X[te] - mu) / sd) @ w + b
        auc = _auc(s_te, y[te])
        # the number that matters: within a single source model, does it separate the
        # prompts that model complied on from the ones it refused on?
        s_all = ((X - mu) / sd) @ w + b
        within = {m: _auc(s_all[(torch.tensor([s == m for s in srcs])) & te],
                          y[(torch.tensor([s == m for s in srcs])) & te])
                  for m in dict.fromkeys(srcs)}
        wv = [v for v in within.values() if v == v]
        wmean = sum(wv) / len(wv) if wv else float("nan")
        # THE decisive number: hold the PROMPT fixed, vary the model. That is the
        # situation a training loss is in — the prompt is given, the weights move.
        # within-model AUC does not control for prompt difficulty (some AdvBench
        # prompts are easy to jailbreak for any model), and L0 — a raw embedding,
        # before any transformer computation — scores nearly as well on it, which
        # proves most of that signal is prompt content, not model state.
        paired_hit = paired_n = 0
        for p in test_p:
            idx = [i for i, q in enumerate(prompts_all) if q == p]
            pos = [i for i in idx if y[i] == 1]
            neg = [i for i in idx if y[i] == 0]
            for a in pos:
                for c in neg:
                    paired_n += 1
                    paired_hit += int(s_all[a] > s_all[c])
        paired = paired_hit / paired_n if paired_n else float("nan")
        print(f"  L{li:<3d} holdout {auc:.3f} | within-model {wmean:.3f} | "
              f"PAIRED(same prompt, diff model) {paired:.3f}")
        if best is None or (paired == paired and paired > best["paired_auc"]):
            best = {"layer": li, "w": w, "b": b, "mu": mu, "sd": sd, "auc": auc,
                    "within_auc": wmean, "within": within, "paired_auc": paired}

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, out)
    print(f"\n[probe] best layer L{best['layer']}: holdout {best['auc']:.3f}, "
          f"within-model {best['within_auc']:.3f}, PAIRED {best['paired_auc']:.3f}")
    print(f"[probe] wrote {out}")
    print("GATE: PAIRED is the one that counts -- same prompt, different model, so prompt\n"
          "difficulty and model identity are both controlled out. ~0.5 means the probe reads\n"
          "the PROMPT, not the model's willingness: useless as a loss, because during training\n"
          "the prompt is fixed and only the weights move. >0.75 means it is reading model state.\n"
          "Compare any layer against L0 (a raw embedding, no transformer computation) -- that is\n"
          "the pure prompt-content baseline.")


if __name__ == "__main__":
    main()
