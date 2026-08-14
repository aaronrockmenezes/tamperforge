"""Per-step view of a training run: the metrics BESIDE the attack that produced them.

    python scripts/tools/va_status.py version_g_gemma31bit_rrcenter    # name under results/
    python scripts/tools/va_status.py results/version_g_gemma31bit_jitter
    python scripts/tools/va_status.py <run> --every 10      # every 10th step
    python scripts/tools/va_status.py <run> --tail 40       # last 40 steps
    python scripts/tools/va_status.py <run> --evals         # clean-probe table too

The run argument is REQUIRED. Auto-picking "the most recent run dir" guesses, and it guessed
wrong the moment two arms trained at once.

L_rr is mean relu(cos) between the ABLATED model and the FROZEN BASE on harmful text, so LOW is
the mechanism working. The header states whether this run used --rr-center, because the
reference values differ and comparing across them is meaningless:

                     uncentred          centred
  Qwen  version_G    0.9854 -> 0.2458   0.9617 -> 0.2072
  gemma version_G    0.9866 -> 0.9522   0.7529 -> 0.3324

gemma's uncentred row is the INSTRUMENT failing, not the mechanism: its residual is 96-99.7% a
shared DC component, which floors the uncentred cosine near 0.96.

Every metric on a row is measured under whatever attack that step SAMPLED, so rows are NOT
comparable across variants -- a coherent harmful preview under surgical means something
different from one under plain. The fixed-attack judged panel is the real gate.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def f(v, spec=".3f", w=7, dash="-"):
    if v is None:
        return f"{dash:>{w}}"
    try:
        return f"{v:>{w}{spec}}"
    except (TypeError, ValueError):
        return f"{str(v):>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name under results/, or a path to the run dir")
    ap.add_argument("--every", type=int, default=1, help="print every Nth step row")
    ap.add_argument("--tail", type=int, default=0, help="only the last N step rows")
    ap.add_argument("--evals", action="store_true", help="also print the clean-probe table")
    a = ap.parse_args()

    d = Path(a.run)
    if not d.exists():
        d = ROOT / "results" / a.run
    if not d.exists():
        sys.exit(f"no such run: {a.run} (looked in ./ and {ROOT / 'results'})")
    ev = d / "events.jsonl"
    if not ev.exists():
        sys.exit(f"no events.jsonl in {d}")

    rows = []
    for line in open(ev):
        line = line.strip()
        if line:
            try:                        # a live run can be mid-write on its last line
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    steps = [r for r in rows if r.get("event") == "step"]
    evals = [r for r in rows if r.get("event") == "eval"]

    cfg = {}
    man = d / "manifest.json"
    if man.exists():
        # write_manifest nests the parsed args one level deeper: config -> {script, run_id, args}.
        c = json.load(open(man)).get("config", {})
        cfg = c.get("args", c) if isinstance(c, dict) else {}

    centred = bool(cfg.get("rr_center"))
    jit = cfg.get("version_b_jitter_deg") or 0
    print(f"{d.name}   {cfg.get('model_id', '?')}  scope={cfg.get('train_scope', '?')}  "
          f"DL={cfg.get('direction_layer', '?')}  lambda_rr={cfg.get('lambda_rr', '?')}"
          f"{'  --rr-center' if centred else ''}{f'  jitter={jit:g}deg' if jit else ''}")
    ref = ("CENTRED (ref: gemma 0.7529->0.3324, qwen 0.9617->0.2072)" if centred
           else "UNCENTRED (ref: gemma 0.9866->0.9522, qwen 0.9854->0.2458)")
    print(f"steps {len(steps)}/{cfg.get('steps', '?')}   evals {len(evals)}   L_rr is {ref}")

    sel = steps[-a.tail:] if a.tail else steps
    sel = sel[::a.every] if a.every > 1 else sel

    hdr = (f"\n{'step':>5} {'L_rr':>7} {'loss':>8} {'L_task':>7} {'refuse':>7} {'ref_abl':>7} "
           f"{'gib_ce':>7} {'gnorm':>8} {'var':>8} {'k':>3} {'ovlap':>6} {'nL':>3} "
           f"{'alpha':>6} {'rdL':>5}  attack_tag")
    print(hdr)
    print("-" * (len(hdr) + 20))
    for r in sel:
        print(f"{r.get('step', ''):>5} {f(r.get('L_rr'), '.4f')} {f(r.get('loss'), '.3f', 8)} "
              f"{f(r.get('L_task'))} {f(r.get('L_safe'))} {f(r.get('ref_abl'))} "
              f"{f(r.get('gib_ce'))} {f(r.get('grad_norm'), '.1f', 8)} "
              f"{str(r.get('attack_variant', '-')):>8} {str(r.get('attack_cap_rank', '-')):>3} "
              f"{f(r.get('attack_cap_overlap'), '.3f', 6)} "
              f"{str(r.get('attack_n_layers', '-')):>3} "
              f"{f(r.get('attack_alpha_mean'), '.2f', 6)} "
              f"{f(r.get('attack_read_layer'), '.1f', 5)}  {r.get('attack_tag', '')}")

    rr = [r["L_rr"] for r in steps if r.get("L_rr") is not None]
    if rr:
        print(f"\nL_rr  first {rr[0]:.4f}  last {rr[-1]:.4f}  min {min(rr):.4f}  n={len(rr)}")
    n = len(steps)
    if n:
        surg = sum(1 for r in steps if r.get("attack_variant") == "surgical")
        wo = sum(1 for r in steps if r.get("attack_write_only"))
        pl = sum(1 for r in steps if r.get("attack_per_layer"))
        print(f"attack mix over {n} steps: surgical {surg / n:.1%}  write-only {wo / n:.1%}  "
              f"per-layer {pl / n:.1%}")

    if a.evals and evals:
        print(f"\n{'step':>5} {'ARC':>6} {'GSM8K':>6} {'IFEval':>6} {'MMLU':>6} "
              f"{'gap_eval':>9} {'L_task_ev':>9} {'L_abl_ev':>9}")
        for r in evals:
            print(f"{r.get('step', ''):>5} {f(r.get('clean_arc_acc'), '.2f', 6)} "
                  f"{f(r.get('clean_gsm8k_acc'), '.2f', 6)} "
                  f"{f(r.get('clean_ifeval_acc'), '.2f', 6)} "
                  f"{f(r.get('clean_mmlu_acc'), '.2f', 6)} "
                  f"{f(r.get('gap_eval'), '.3f', 9)} {f(r.get('L_task_eval'), '.3f', 9)} "
                  f"{f(r.get('L_abl_eval'), '.3f', 9)}")
        print("GSM8K here is the in-loop CLEAN probe on the TRAIN split -- a trend, never a score.")


if __name__ == "__main__":
    main()
