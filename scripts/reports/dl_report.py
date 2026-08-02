import json, glob, os
cells = [("L8","DL 8   read+write"),("L12","DL 12  read+write"),("L14","DL 14  read+write"),
         ("L16","DL 16  read+write"),("L20","DL 20  read+write  <-CONTROL"),
         ("L24","DL 24  read+write"),("L26","DL 26  read+write"),
         ("L14wo","DL 14  WRITE-ONLY"),("L20wo","DL 20  WRITE-ONLY")]
print(f"{'cell':<28}{'harmful':>9}{'gibber':>8}{'refused':>9}{'GSM8K':>8}")
print("-"*62)
for tag,label in cells:
    p=f"results/dlsw_{tag}_judged/summary.json"
    if os.path.exists(p):
        s=json.load(open(p)); u=s["usefulness_counts"]; n=s["n"]
        h,g,r=(u.get("harmful_actionable",0)/n,u.get("gibberish",0)/n,u.get("refused",0)/n)
    else:
        print(f"{label:<28}{'  ...pending':>9}"); continue
    gs=float("nan")
    for f in glob.glob(f"results/dlswcap_{tag}_gsm8k/**/results_*.json",recursive=True):
        try: d=json.load(open(f))
        except Exception: continue
        for k,v in d.get("results",{}).get("gsm8k",{}).items():
            if k.startswith("exact_match,") and "flexible" in k: gs=v
    print(f"{label:<28}{h:>9.4f}{g:>8.4f}{r:>9.4f}{gs:>8.4f}")
print(f"\n{'clean reference':<28}{0.0058:>9.4f}{0.0077:>8.4f}{0.9865:>9.4f}{0.3813:>8.4f}")
print(f"{'heretic t175 (worst)':<28}{0.2615:>9.4f}{0.5942:>8.4f}{0.0712:>9.4f}{0.3844:>8.4f}")
