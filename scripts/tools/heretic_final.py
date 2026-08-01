import json, glob, os, statistics as st
def cap(tag):
    o = {}
    for f in glob.glob(f"results/hvacap_{tag}_*/**/results_*.json", recursive=True):
        try: d = json.load(open(f))
        except Exception: continue
        for task, m in d.get("results", {}).items():
            for k, v in m.items():
                if not isinstance(v, float): continue
                if k.startswith("acc,"): o.setdefault(task, {})["acc"] = v
                elif k.startswith("exact_match,") and "flexible" in k: o.setdefault(task, {})["em"] = v
    mm = [m["acc"] for t, m in o.items() if t.startswith("mmlu_") and "acc" in m]
    return (o.get("arc_challenge", {}).get("acc", float("nan")),
            st.mean(mm) if mm else float("nan"),
            o.get("gsm8k", {}).get("em", float("nan")))
def saf(tag):
    p = f"results/hva_{tag}_judged/summary.json"
    if not os.path.exists(p): return (float("nan"),)*3
    s = json.load(open(p)); u = s["usefulness_counts"]; n = s["n"]
    return (u.get("harmful_actionable",0)/n, u.get("gibberish",0)/n, u.get("refused",0)/n)
print(f"{'attack on version_A s500':<30}{'harmful':>9}{'gibber':>8}{'refused':>9}{'ARC':>8}{'MMLU':>8}{'GSM8K':>8}")
print("-"*80)
print(f"{'clean (no attack)':<30}{0.0058:>9.4f}{0.0077:>8.4f}{0.9865:>9.4f}{0.3063:>8.4f}{0.4193:>8.4f}{0.3813:>8.4f}")
print(f"{'rank-1 ablation':<30}{0.0000:>9.4f}{1.0000:>8.4f}{0.0000:>9.4f}{0.2688:>8.4f}{0.3278:>8.4f}{0.0394:>8.4f}")
print(f"{'surgical k16':<30}{0.0019:>9.4f}{0.0269:>8.4f}{0.9712:>9.4f}{0.3029:>8.4f}{0.4281:>8.4f}{0.3738:>8.4f}")
for t, lbl in [("t85","heretic t85  KL .0278"),("t175","heretic t175 KL .0124"),("t100","heretic t100 KL .0092")]:
    h,g,r = saf(t); a,m,gs = cap(t)
    print(f"{lbl:<30}{h:>9.4f}{g:>8.4f}{r:>9.4f}{a:>8.4f}{m:>8.4f}{gs:>8.4f}")
print("\nv8 reference: heretic = 0.82 harmful (campaign), surgical k16 = 0.4365, clean = 0.9827 refused")
