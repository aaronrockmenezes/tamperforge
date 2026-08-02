import json, sys, glob, os
d = sys.argv[1]
s = json.load(open(f"{d}/summary.json"))
bc = s.get("by_condition", {})
print(f"n={s['n']}  judge={s['judge_model']}\n")
hdr = f"{'condition':<34} {'n':>3} {'harmful':>8} {'gibber':>8} {'refused':>8} {'benign':>7} {'ASR':>6}"
print(hdr); print("-"*len(hdr))
def row(name, c):
    u = c.get("usefulness_counts", {}); n = c.get("n", 0) or 1
    return (f"{name:<34} {c.get('n',0):>3} "
            f"{u.get('harmful_actionable',0)/n:>8.3f} {u.get('gibberish',0)/n:>8.3f} "
            f"{u.get('refused',0)/n:>8.3f} {u.get('benign',0)/n:>7.3f} {c.get('judge_asr',0):>6.3f}")
order = ["clean","canonical_rank1","heretic_perlayer_a04","heretic_perlayer_a08",
         "surgical_k4","surgical_k16","surgical_k16_perlayer"]
keys = sorted(bc, key=lambda k: next((i for i,o in enumerate(order) if k.endswith("::"+o)), 99))
for k in keys:
    print(row(k.split("::")[-1], bc[k]))
