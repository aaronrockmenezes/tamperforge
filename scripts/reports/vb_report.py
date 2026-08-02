import json, glob, os, statistics as st
def cap(tag):
    o={}
    for f in glob.glob(f"results/xvbcap_{tag}_*/**/results_*.json", recursive=True):
        try: d=json.load(open(f))
        except Exception: continue
        for task,m in d.get("results",{}).items():
            for k,v in m.items():
                if not isinstance(v,float): continue
                if k.startswith("acc,"): o.setdefault(task,{})["acc"]=v
                elif k.startswith("exact_match,") and "flexible" in k: o.setdefault(task,{})["em"]=v
    mm=[m["acc"] for t,m in o.items() if t.startswith("mmlu_") and "acc" in m]
    return (o.get("arc_challenge",{}).get("acc",float("nan")),
            st.mean(mm) if mm else float("nan"),
            o.get("gsm8k",{}).get("em",float("nan")))
def saf(tag):
    p=f"results/xvb_{tag}_judged/summary.json"
    if not os.path.exists(p): return None
    s=json.load(open(p)); u=s["usefulness_counts"]; n=s["n"]
    return (u.get("harmful_actionable",0)/n,u.get("gibberish",0)/n,u.get("refused",0)/n)
A={"clean":(0.0058,0.0077,0.9865,0.3063,0.4193,0.3813),
   "rank1":(0.0000,1.0000,0.0000,0.2688,0.3278,0.0394),
   "surg_k16":(0.0019,0.0269,0.9712,0.3029,0.4281,0.3738)}
print(f"{'':<12}{'harmful':>9}{'gibber':>8}{'refused':>9}{'ARC':>8}{'MMLU':>8}{'GSM8K':>8}")
print("-"*62)
for tag in ["clean","rank1","surg_k16"]:
    s=saf(tag); a,m,g=cap(tag)
    if s is None: print(f"{'B '+tag:<12}{'  running':>9}"); 
    else: print(f"{'B '+tag:<12}{s[0]:>9.4f}{s[1]:>8.4f}{s[2]:>9.4f}{a:>8.4f}{m:>8.4f}{g:>8.4f}")
    r=A[tag]; print(f"{'A '+tag:<12}{r[0]:>9.4f}{r[1]:>8.4f}{r[2]:>9.4f}{r[3]:>8.4f}{r[4]:>8.4f}{r[5]:>8.4f}")
    print()
