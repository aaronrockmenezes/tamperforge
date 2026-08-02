import re, statistics as st
def parse(path):
    txt = open(path, encoding="utf-8", errors="replace").read().replace("\r","\n")
    blocks = re.split(r"Running trial (\d+) of \d+", txt)
    out=[]
    for i in range(1,len(blocks)-1,2):
        tid,body=int(blocks[i]),blocks[i+1]
        d={"trial":tid}
        for m in re.finditer(r"\*\s+([a-zA-Z0-9_.]+)\s*=\s*(-?[\d.]+)", body): d[m.group(1)]=float(m.group(2))
        kl=re.search(r"KL divergence:\s*([\d.]+)",body); rf=re.search(r"Refusals:\s*(\d+)/",body)
        if kl: d["kl"]=float(kl.group(1))
        if rf: d["refusals"]=int(rf.group(1))
        out.append(d)
    return [t for t in out if "refusals" in t and "direction_index" in t]
for path,lbl in [("logs/training_runs/heretic_version_a_s500.log","version_A"),
                 ("logs/training_runs/heretic_version_b.log","version_B")]:
    try: ts=parse(path)
    except FileNotFoundError: print(f"{lbl}: log missing"); continue
    best=sorted(ts,key=lambda t:(t["refusals"],t["kl"]))[:8]
    di=[t["direction_index"] for t in best]
    print(f"{lbl}: top-8 trials by fewest refusals")
    print(f"   direction_index  min {min(di):.2f}  median {st.median(di):.2f}  max {max(di):.2f}")
    print("   " + "  ".join(f"t{t['trial']}:{t['direction_index']:.1f}(r{t['refusals']})" for t in best[:6]))
