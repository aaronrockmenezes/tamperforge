"""Gate two FORM-based coherence detectors, as candidate gib_ce replacements.

The premise: gibberish-vs-coherent is a question about the SHAPE of the text, not its
meaning, so it should be measurable without any reference string and without semantics.
Every likelihood-based attempt has failed on exactly this point -- degenerate repetition
is the most probable text there is, so likelihood scores the collapse mode as excellent
(gib_ce 0.09-0.84 nats across four judge categories; option D's frozen scorer came out
sign-inverted at AUC 0.06-0.21).

Two candidates, both differentiable, neither needing generation or a reference text:

  erank   Effective rank of the residual-stream trajectory over response positions.
          Coherent text moves through representation space -> many directions used ->
          high effective rank. A repetition loop stops moving -> the trajectory
          collapses toward a line -> rank ~1. Computed as exp(entropy of the
          normalised singular value spectrum). Also reports top1, the share of
          spectral mass in the leading singular value (high = collapsed).

  repmass Probability the model puts on tokens it recently emitted:
          mean_t sum_{k=1..W} p_t[token_{t-k}]. High = the model wants to loop. This
          is the differentiable form of an n-gram repetition rate.

Scored on generations that are already on disk and already judged, entirely WITHIN each
checkpoint, so model identity cannot leak in the way it did for the option-F probe.

GATE: AUC(gibberish vs coherent) per metric. gib_ce's equivalent separation is
0.09-0.84 nats, i.e. near-chance. Anything reaching ~0.9 here is a real replacement --
and a simpler loss than the harm_ce + L_rr pair currently carrying that job.

  python experiments/gate_form_metrics.py --runs \
    v8_clean=results/t0v2_qwen_v8_clean \
    rank1=results/t0v2_qwen_v8_att_rank1 \
    heretic_strong=results/t0v2_heretic_kl2222_trial160
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

MODEL_OF = {
    "v8_clean": "outputs/qwen_v8_clean",
    "rank1": "outputs/qwen_v8_att_rank1",
    "heretic_weak": "outputs/attacked_snapshots/heretic_qwen_v8_kl0042_trial192",
    "heretic_mid": "outputs/attacked_snapshots/heretic_qwen_v8_kl0778_trial91",
    "heretic_strong": "outputs/attacked_snapshots/heretic_qwen_v8_kl2222_trial160",
}


def _auc(s: list[float], y: list[int]) -> float:
    t = torch.tensor(s)
    yy = torch.tensor(y)
    pos, neg = t[yy == 1], t[yy == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = torch.cat([pos, neg]).argsort().argsort().float()
    return ((r[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))).item()


def _rows(run: Path) -> list[dict]:
    gens = {json.loads(x)["prompt"]: json.loads(x) for x in open(run / "generations.jsonl")}
    out = []
    for line in open(run.parent / (run.name + "_judged") / "judgments.jsonl"):
        r = json.loads(line)
        j = r.get("judge")
        if isinstance(j, str):
            j = ast.literal_eval(j)
        p = (j or {}).get("parsed") or {}
        if p.get("refusal") is None:
            continue
        g = gens.get(r["prompt"])
        if not g or not (g.get("response") or "").strip():
            continue
        out.append({"prompt": r["prompt"], "response": g["response"],
                    "label": usefulness_label(p)})
    return out


@torch.no_grad()
def _metrics(model, tok, device, prompt: str, response: str, window: int = 8):
    prefix = apply_chat_template_no_think(tok, [{"role": "user", "content": prompt}],
                                          tokenize=False, add_generation_prompt=True)
    enc = tok(prefix + response, return_tensors="pt", truncation=True, max_length=768).to(device)
    plen = min(tok(prefix, return_tensors="pt")["input_ids"].shape[1],
               enc["input_ids"].shape[1] - 2)
    ids = enc["input_ids"][0]
    if ids.shape[0] - plen < 8:
        return None
    out = model(**enc, output_hidden_states=True)

    # --- repmass: mass the model puts on tokens it recently emitted ---
    logits = out.logits[0].float()
    p = torch.softmax(logits, dim=-1)
    tot = 0.0
    n = 0
    for t in range(plen, ids.shape[0] - 1):
        prev = ids[max(0, t - window + 1): t + 1]
        tot += float(p[t, prev].sum())
        n += 1
    repmass = tot / max(n, 1)

    # --- erank / top1: does the residual trajectory keep moving across positions? ---
    per_layer = {}
    for li, h in enumerate(out.hidden_states):
        H = h[0, plen:, :].float()
        H = H - H.mean(0, keepdim=True)
        H = H / H.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        sv = torch.linalg.svdvals(H)
        q = sv / sv.sum().clamp(min=1e-9)
        erank = float(torch.exp(-(q * (q + 1e-12).log()).sum()))
        per_layer[li] = (erank, float(q[0]))
    return repmass, per_layer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, metavar="LABEL=RESULTS_DIR")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/gate_form_metrics.json")
    args = ap.parse_args()

    report = {}
    for spec in args.runs:
        label, rdir = spec.split("=", 1)
        model, tok, device = load_model(MODEL_OF[label], device=args.device)
        rows = _rows(ROOT / rdir)
        if args.limit > 0:
            rows = rows[: args.limit]
        keep = []
        for r in rows:
            m = _metrics(model, tok, device, r["prompt"], r["response"])
            if m is None:
                continue
            r["repmass"], r["per_layer"] = m
            keep.append(r)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        y = [int(r["label"] == "gibberish") for r in keep]
        coh = [r["label"] in ("harmful_actionable", "refused", "benign") for r in keep]
        idx = [i for i in range(len(keep)) if y[i] or coh[i]]
        yy = [y[i] for i in idx]
        n_gib, n_coh = sum(yy), len(yy) - sum(yy)
        print(f"\n=== {label}: {len(keep)} rows ({n_gib} gibberish / {n_coh} coherent) ===")
        if n_gib < 5 or n_coh < 5:
            print("  too few of one class -- skipping AUC")
            continue

        auc_rep = _auc([keep[i]["repmass"] for i in idx], yy)
        gm = [keep[i]["repmass"] for i in idx if keep[i]["label"] == "gibberish"]
        cm = [keep[i]["repmass"] for i in idx if keep[i]["label"] != "gibberish"]
        print(f"  repmass  gib {sum(gm)/len(gm):.4f} vs coherent {sum(cm)/len(cm):.4f}"
              f"   AUC {auc_rep:.3f}")

        # Full per-layer profile. L0 is the raw embedding, so erank/top1 there is a
        # type-token ratio (lexical repetition) and needs no transformer computation --
        # it is the baseline every deeper layer must beat to claim a representational
        # signal rather than a surface one.
        nlay = len(keep[0]["per_layer"])
        prof = {}
        print("  layer:  erank / top1   (erank <0.5 = gibberish collapses, as predicted)")
        for li in range(nlay):
            e = _auc([keep[i]["per_layer"][li][0] for i in idx], yy)
            t = _auc([keep[i]["per_layer"][li][1] for i in idx], yy)
            prof[li] = {"auc_erank": e, "auc_top1": t}
            tag = "  <- L0 lexical baseline" if li == 0 else ""
            print(f"   L{li:<3d}  {e:.3f} / {t:.3f}{tag}")
        best_e = max(range(nlay), key=lambda li: abs(prof[li]["auc_erank"] - 0.5))
        report[label] = {"n": len(keep), "n_gib": n_gib, "n_coh": n_coh,
                         "auc_repmass": auc_rep, "repmass_gib": sum(gm) / len(gm),
                         "repmass_coh": sum(cm) / len(cm),
                         "erank_best_layer": best_e, "per_layer": prof}

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[form] wrote {out}")
    print("GATE: AUC near 1.0 (or near 0.0 -- a clean inversion is just as usable, flip the "
          "sign) means the metric separates collapse from real text. gib_ce's equivalent is "
          "near-chance. AUC ~0.5 means dead.")


if __name__ == "__main__":
    main()
