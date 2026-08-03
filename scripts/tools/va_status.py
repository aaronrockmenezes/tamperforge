"""Joined per-eval view of a version_A run: metrics BESIDE the attack that produced them.

The training log prints an AdvBench preview under whatever attack that step sampled, but
never says which one. With version_A drawing surgical ~33% of the time, a coherent harmful
preview is ambiguous -- unformed wall, or a wall that simply does not cover surgical. This
joins each eval to its attack so the two are distinguishable.
"""
import glob
import json
import os
import sys
from pathlib import Path

# Resolve against the repo root, not cwd -- this globbed relative paths and blew up with
# "max() iterable argument is empty" whenever run from anywhere but the root.
ROOT = Path(__file__).resolve().parents[2]


def fmt(v, spec=".3f", width=7, dash="-"):
    if v is None:
        return f"{dash:>{width}}"
    try:
        return f"{v:>{width}{spec}}"
    except (TypeError, ValueError):
        return f"{str(v):>{width}}"


if len(sys.argv) > 1:
    d = sys.argv[1]
else:
    cands = glob.glob(str(ROOT / "results" / "tamper_resistant_p1b_*"))
    if not cands:
        sys.exit(f"no runs found under {ROOT / 'results'} -- pass a run dir explicitly")
    d = max(cands, key=os.path.getmtime)

ev = Path(d) / "events.jsonl"
if not ev.exists():
    sys.exit(f"no events.jsonl in {d}")
rows = [json.loads(l) for l in open(ev) if l.strip()]
rows = [r for r in rows if "step" in r]
print(f"run: {d}   evals: {len(rows)}\n")

hdr = (f"{'step':>5} {'variant':>9} {'k':>3} {'overlap':>7} {'gib_ce':>8} "
       f"{'gap_eval':>9} {'IFEval':>7} {'GSM8K':>7} {'refuse_abl':>10}  attack_tag")
print(hdr)
print("-" * (len(hdr) + 18))
for r in rows:
    print(f"{r.get('step', ''):>5} "
          f"{str(r.get('attack_variant', '-')):>9} "
          f"{str(r.get('attack_cap_rank', '-')):>3} "
          f"{fmt(r.get('attack_cap_overlap'))} "
          f"{fmt(r.get('gib_ce'), width=8)} "
          f"{fmt(r.get('gap_eval'), width=9)} "
          f"{fmt(r.get('clean_ifeval_acc'), spec='.2f')} "
          f"{fmt(r.get('clean_gsm8k_acc'), spec='.2f')} "
          f"{fmt(r.get('ref_abl'), spec='.2f', width=10)}  "
          f"{r.get('attack_tag', '')}")

surg = [r for r in rows if r.get("attack_variant") == "surgical"]
plain = [r for r in rows if r.get("attack_variant") == "plain"]
print(f"\nevals under surgical: {len(surg)}   under plain: {len(plain)}")
print("NOTE: gib_ce/gap_eval here are measured under the SAMPLED attack, so rows are not")
print("comparable across variants. The fixed-attack judged panel is the real gate.")
print("GSM8K = in-loop CLEAN probe (--gsm8k-probe-n), GSM8K TRAIN split, flexible-extract")
print("(last number). A TREND, not a score -- never quote it as a GSM8K result. Blank if the")
print("run predates the probe or was launched without --gsm8k-probe-n.")
gsm = [r.get("clean_gsm8k_acc") for r in rows if r.get("clean_gsm8k_acc") is not None]
if gsm:
    print(f"clean GSM8K probe: first {gsm[0]:.2f} -> last {gsm[-1]:.2f} "
          f"(min {min(gsm):.2f}, max {max(gsm):.2f}, n={len(gsm)})")
