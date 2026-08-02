import json, glob, os
def grab(tag):
    out = {}
    for f in glob.glob(f"results/vacap_{tag}_*/**/results_*.json", recursive=True):
        try: d = json.load(open(f))
        except Exception: continue
        for task, m in d.get("results", {}).items():
            for k, v in m.items():
                if not isinstance(v, float): continue
                if k.startswith("acc,"): out.setdefault(task, {})["acc"] = v
                elif k.startswith("acc_norm,"): out.setdefault(task, {})["acc_norm"] = v
                elif k.startswith("exact_match,") and "flexible" in k: out.setdefault(task, {})["em"] = v
    return out
def mmlu_avg(o):
    v = [m["acc"] for t, m in o.items() if t.startswith("mmlu_") and "acc" in m]
    return sum(v)/len(v) if v else float("nan")
print(f"{'variant':<26}{'ARC acc':>9}{'ARC_n':>8}{'MMLU(12)':>10}{'GSM8K':>8}")
print("-"*61)
for tag, label in [("va_clean","version_A clean"),("va_rank1","version_A + rank-1"),
                   ("va_surg_k16","version_A + surgical k16")]:
    o = grab(tag); a = o.get("arc_challenge", {}); g = o.get("gsm8k", {})
    print(f"{label:<26}{a.get('acc',float('nan')):>9.4f}{a.get('acc_norm',float('nan')):>8.4f}"
          f"{mmlu_avg(o):>10.4f}{g.get('em',float('nan')):>8.4f}")
print()
print("v8 campaign reference (same lm_eval config):")
print(f"{'  v8 clean':<26}{0.3217:>9.4f}{0.3447:>8.4f}{0.437:>10.4f}{0.3889:>8.4f}")
print(f"{'  v8 + rank-1 (surg k0)':<26}{0.2218:>9.4f}{0.2696:>8.4f}{0.253:>10.4f}{0.0212:>8.4f}")
print(f"{'  v8 + surgical k16':<26}{0.2969:>9.4f}{0.3345:>8.4f}{0.401:>10.4f}{0.2282:>8.4f}")
