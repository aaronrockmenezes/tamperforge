import json, glob, os
print(f"{'alpha':>6}{'harmful':>10}{'gibber':>9}{'refused':>9}{'GSM8K':>9}")
print("-"*43)
for a in ["0.2","0.4","0.6","0.8","1.0","1.2","1.5"]:
    tag = "a" + a.replace(".", "")
    p = f"results/asw_{tag}_judged/summary.json"
    if os.path.exists(p):
        s = json.load(open(p)); u = s["usefulness_counts"]; n = s["n"]
        h, g, r = (u.get("harmful_actionable",0)/n, u.get("gibberish",0)/n, u.get("refused",0)/n)
    else:
        h = g = r = float("nan")
    gs = float("nan")
    for f in glob.glob(f"results/aswcap_{tag}_gsm8k/**/results_*.json", recursive=True):
        try: d = json.load(open(f))
        except Exception: continue
        for k, v in d.get("results", {}).get("gsm8k", {}).items():
            if k.startswith("exact_match,") and "flexible" in k: gs = v
    star = "  <- control (rank-1)" if a == "1.0" else ""
    print(f"{a:>6}{h:>10.4f}{g:>9.4f}{r:>9.4f}{gs:>9.4f}{star}")
print(f"\n{'clean':>6}{0.0058:>10.4f}{0.0077:>9.4f}{0.9865:>9.4f}{0.3813:>9.4f}")
print(f"{'known rank-1 row: 0.0000 harmful / 1.0000 gib / GSM8K 0.0394':>6}")
