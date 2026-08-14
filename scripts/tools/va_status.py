"""Live status for one or more training runs: L_rr trajectory, loss terms, and the attack mix.

WHY IT LOOKS LIKE THIS. Two things this had to fix.

  * L_rr was not reported anywhere. Its absence cost a full day of gemma forensics: it lived
    only in events.jsonl, gemma's training events.jsonl was never archived, and four
    architectural hypotheses were chased before anyone could check whether the mechanism being
    credited had converged. It is now the FIRST thing printed.
  * The old version printed one row per event and filtered on `"step" in r`, which matches
    every step row as well as every eval row -- 500 lines for a 500-step run, with the three
    eval rows buried inside. events.jsonl actually carries three event types: `step` (per-step
    loss terms + the attack that produced them), `eval` (periodic clean probes), and
    `advbench_preview`. They are now read separately.

Reading L_rr: it is mean relu(cos) between the ABLATED model and the FROZEN BASE on harmful
text, so LOW is the mechanism working. Reference values from results/posthoc_lrr.json:

                     uncentred          centred
  Qwen  version_G    0.9854 -> 0.2458   0.9617 -> 0.2072
  gemma version_G    0.9866 -> 0.9522   0.7529 -> 0.3324

gemma's uncentred row is the INSTRUMENT failing, not the mechanism -- its residual is 96-99.7%
a shared DC component, which floors the uncentred cosine near 0.96. Compare a --rr-center run
against the centred column, never the uncentred one.

  python scripts/tools/va_status.py                 # the 2 most recent runs
  python scripts/tools/va_status.py results/tamper_resistant_p1b_XXXX [more...]
  python scripts/tools/va_status.py --evals         # also the periodic clean-probe table
"""
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fmt(v, spec=".3f", width=7, dash="-"):
    if v is None:
        return f"{dash:>{width}}"
    try:
        return f"{v:>{width}{spec}}"
    except (TypeError, ValueError):
        return f"{str(v):>{width}}"


def load(d):
    ev = Path(d) / "events.jsonl"
    if not ev.exists():
        return None
    rows = []
    for line in open(ev):
        line = line.strip()
        if not line:
            continue
        try:                                   # a live run can be mid-write on the last line
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    cfg = {}
    man = Path(d) / "manifest.json"
    if man.exists():
        # write_manifest nests the parsed args one level deeper: config -> {script, run_id, args}.
        # Reading config directly yields no flags at all, so every arm silently rendered as
        # "UNCENTRED lambda_rr=?" -- including the --rr-center arm, whose whole point is that it
        # must be compared against the CENTRED reference column.
        c = json.load(open(man)).get("config", {})
        cfg = c.get("args", c) if isinstance(c, dict) else {}
    return {"dir": str(d), "cfg": cfg,
            "steps": [r for r in rows if r.get("event") == "step"],
            "evals": [r for r in rows if r.get("event") == "eval"]}


def arm_label(cfg):
    bits = []
    if cfg.get("rr_center"):
        bits.append("--rr-center")
    j = cfg.get("version_b_jitter_deg") or 0
    if j:
        bits.append(f"jitter={j:g}deg")
    bits.append(f"lambda_rr={cfg.get('lambda_rr', '?')}")
    return " ".join(bits)


def spark(vals, n=12):
    """Evenly-spaced samples across the run -- a trajectory, not a final number."""
    if not vals:
        return "-"
    if len(vals) <= n:
        sel = vals
    else:
        step = (len(vals) - 1) / (n - 1)
        sel = [vals[round(i * step)] for i in range(n)]
    return " ".join(f"{v:.3f}" for v in sel)


def report(run, show_evals):
    st, ev, cfg = run["steps"], run["evals"], run["cfg"]
    out = cfg.get("out") or "?"
    print(f"\n{'=' * 100}\n{Path(run['dir']).name}   {Path(str(out)).name}")
    print(f"  {arm_label(cfg)}   model={cfg.get('model_id', '?')}  "
          f"scope={cfg.get('train_scope', '?')}  DL={cfg.get('direction_layer', '?')}")
    if not st:
        print("  no step rows yet")
        return
    last = st[-1]
    print(f"  steps logged: {len(st)} / {cfg.get('steps', '?')}    "
          f"last step {last.get('step')}")

    rr = [r["L_rr"] for r in st if r.get("L_rr") is not None]
    if rr:
        centred = " (centred -- compare vs 0.7529->0.3324 gemma / 0.9617->0.2072 qwen)" \
            if cfg.get("rr_center") else " (UNCENTRED -- DC-floored on gemma, see docstring)"
        print(f"\n  L_rr{centred}")
        print(f"    trajectory: {spark(rr)}")
        print(f"    first {rr[0]:.4f}   last {rr[-1]:.4f}   min {min(rr):.4f}   "
              f"moved {rr[0] - rr[-1]:+.4f} ({(rr[0] - rr[-1]) / max(rr[0], 1e-9) * 100:.1f}% of start)")

    print("\n  loss terms @ last step")
    terms = [("loss", "loss"), ("L_task", "L_task"), ("L_safe", "refuse clean"),
             ("ref_abl", "refuse ablated"), ("L_rr", "L_rr"), ("L_harm", "L_harm"),
             ("gib_ce", "gib_ce"), ("L_clean_gen", "clean_gen_KL"), ("grad_norm", "grad_norm")]
    print("    " + "  ".join(f"{lab}={fmt(last.get(k), width=0)}" for k, lab in terms))

    # The axis version_A exists to vary. A wall that never saw surgical cannot cover it.
    n = len(st)
    surg = [r for r in st if r.get("attack_variant") == "surgical"]
    wo = [r for r in st if r.get("attack_write_only")]
    pl = [r for r in st if r.get("attack_per_layer")]
    ov = [r["attack_cap_overlap"] for r in st if isinstance(r.get("attack_cap_overlap"), (int, float))]
    print(f"\n  attack mix over {n} steps")
    print(f"    surgical {len(surg) / n:.1%}   write-only {len(wo) / n:.1%}   "
          f"per-layer {len(pl) / n:.1%}")
    ks = {}
    for r in surg:
        ks[r.get("attack_cap_rank")] = ks.get(r.get("attack_cap_rank"), 0) + 1
    if ks:
        print(f"    cap_rank: " + "  ".join(f"k={k}:{v}" for k, v in sorted(ks.items(), key=lambda x: (x[0] is None, x[0]))))
    if ov:
        print(f"    cap_overlap: mean {sum(ov) / len(ov):.3f}  min {min(ov):.3f}  max {max(ov):.3f}")
    lo = [r["attack_layer_min"] for r in st if r.get("attack_layer_min") is not None]
    hi = [r["attack_layer_max"] for r in st if r.get("attack_layer_max") is not None]
    nl = [r["attack_n_layers"] for r in st if r.get("attack_n_layers") is not None]
    if lo and hi:
        print(f"    layers: span {min(lo)}..{max(hi)}   mean n_layers {sum(nl) / len(nl):.1f}")
    al = [r["attack_alpha_mean"] for r in st if r.get("attack_alpha_mean") is not None]
    if al:
        print(f"    alpha: mean {sum(al) / len(al):.3f}  max {max(al):.3f}")
    rl = [r["attack_read_layer"] for r in st if r.get("attack_read_layer") is not None]
    if rl:
        print(f"    read_layer: {min(rl):.1f}..{max(rl):.1f}  mean {sum(rl) / len(rl):.2f}")

    if ev:
        print(f"\n  clean probes ({len(ev)} evals)  -- GSM8K is an in-loop TREND, never a score")
        h = (f"    {'step':>5} {'ARC':>6} {'GSM8K':>6} {'IFEval':>6} {'MMLU':>6} "
             f"{'gap_eval':>9} {'L_task_ev':>9}")
        print(h)
        for r in ev if show_evals else ev[-6:]:
            print(f"    {r.get('step', ''):>5} {fmt(r.get('clean_arc_acc'), '.2f', 6)} "
                  f"{fmt(r.get('clean_gsm8k_acc'), '.2f', 6)} "
                  f"{fmt(r.get('clean_ifeval_acc'), '.2f', 6)} "
                  f"{fmt(r.get('clean_mmlu_acc'), '.2f', 6)} "
                  f"{fmt(r.get('gap_eval'), '.3f', 9)} {fmt(r.get('L_task_eval'), '.3f', 9)}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    show_evals = "--evals" in sys.argv
    if args:
        dirs = args
    else:
        # Any results dir holding an events.jsonl. Globbing tamper_resistant_p1b_* missed every
        # run launched with an explicit --run-id, which is now all of them: the trainer defaults
        # run_id to a timestamp, so two arms started in the same second collide into one dir.
        cands = [d for d in glob.glob(str(ROOT / "results" / "*"))
                 if os.path.isfile(os.path.join(d, "events.jsonl"))]
        if not cands:
            sys.exit(f"no runs with events.jsonl under {ROOT / 'results'} -- pass a dir explicitly")
        dirs = sorted(cands, key=os.path.getmtime, reverse=True)[:2]
    runs = [r for r in (load(d) for d in dirs) if r]
    if not runs:
        sys.exit("no events.jsonl in any given run dir")
    for r in runs:
        report(r, show_evals)

    rr = [(Path(r["dir"]).name, r["cfg"].get("rr_center"),
           [x["L_rr"] for x in r["steps"] if x.get("L_rr") is not None]) for r in runs]
    rr = [(n, c, v) for n, c, v in rr if v]
    if len(rr) > 1:
        print(f"\n{'=' * 100}\nL_rr side by side (LOW = rerouting working)")
        for n, c, v in rr:
            print(f"  {n:34s} {'centred  ' if c else 'uncentred'} "
                  f"first {v[0]:.4f}  last {v[-1]:.4f}  min {min(v):.4f}  n={len(v)}")
        print("  Only compare like with like: an uncentred L_rr is floored near 0.96 on gemma.")


if __name__ == "__main__":
    main()
