import re, statistics as st
txt = open("logs/training_runs/heretic_version_a_s500.log", encoding="utf-8", errors="replace").read()
txt = txt.replace("\r", "\n")
# each trial block: params then KL then refusals
blocks = re.split(r"Running trial (\d+) of \d+", txt)
trials = []
for i in range(1, len(blocks)-1, 2):
    tid, body = int(blocks[i]), blocks[i+1]
    d = {"trial": tid}
    for m in re.finditer(r"\*\s+([a-zA-Z0-9_.]+)\s*=\s*(-?[\d.]+)", body):
        d[m.group(1)] = float(m.group(2))
    kl = re.search(r"KL divergence:\s*([\d.]+)", body)
    rf = re.search(r"Refusals:\s*(\d+)/(\d+)", body)
    if kl: d["kl"] = float(kl.group(1))
    if rf: d["refusals"] = int(rf.group(1))
    trials.append(d)
print(f"parsed trials: {len(trials)}\n")
keys = ["direction_index","attn.o_proj.max_weight","attn.o_proj.max_weight_position",
        "attn.o_proj.min_weight","attn.o_proj.min_weight_distance",
        "mlp.down_proj.max_weight","mlp.down_proj.max_weight_position",
        "mlp.down_proj.min_weight","mlp.down_proj.min_weight_distance"]
print(f"{'parameter':<40}{'min':>8}{'med':>8}{'max':>8}{'n>1.0':>7}{'n>2.0':>7}")
print("-"*78)
for k in keys:
    v = [t[k] for t in trials if k in t]
    if not v: continue
    print(f"{k:<40}{min(v):>8.2f}{st.median(v):>8.2f}{max(v):>8.2f}"
          f"{sum(x>1.0 for x in v):>7}{sum(x>2.0 for x in v):>7}")
best = sorted([t for t in trials if "refusals" in t], key=lambda t: (t["refusals"], t["kl"]))[:6]
print("\ntop trials by fewest refusals:")
print(f"{'trial':>6}{'refus':>7}{'KL':>9}{'dir_idx':>9}{'o.max':>7}{'o.min':>7}{'dn.max':>8}{'dn.min':>8}")
for t in best:
    print(f"{t['trial']:>6}{t['refusals']:>7}{t['kl']:>9.4f}{t.get('direction_index',float('nan')):>9.2f}"
          f"{t.get('attn.o_proj.max_weight',float('nan')):>7.2f}{t.get('attn.o_proj.min_weight',float('nan')):>7.2f}"
          f"{t.get('mlp.down_proj.max_weight',float('nan')):>8.2f}{t.get('mlp.down_proj.min_weight',float('nan')):>8.2f}")
