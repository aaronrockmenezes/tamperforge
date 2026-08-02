import json, sys
d = sys.argv[1]
s = json.load(open(f"{d}/summary.json"))
bc = s.get("by_condition", {})
order = ["clean","canonical_rank1","heretic_perlayer_a04","heretic_perlayer_a08",
         "surgical_k4","surgical_k16","surgical_k16_perlayer"]
def ck(k): return k.split("::")[0]
def cond(k): return k.split("::")[-1]
cks = sorted({ck(k) for k in bc})
print(f"n={s['n']}  parse_failures={s.get('parse_failures')}  judge={s['judge_model']}\n")
hdr = f"{'condition':<24}" + "".join(f"{c[-8:]:>26}" for c in cks)
print(hdr); print("-"*len(hdr))
print(f"{'':<24}" + "".join(f"{'harm/gib/ref':>26}" for _ in cks))
for o in order:
    line = f"{o:<24}"
    for c in cks:
        k = next((k for k in bc if ck(k)==c and cond(k)==o), None)
        if k is None: line += f"{'-':>26}"; continue
        u = bc[k].get("usefulness_counts",{}); n = bc[k].get("n",1) or 1
        line += f"{u.get('harmful_actionable',0)/n:>10.3f}{u.get('gibberish',0)/n:>8.3f}{u.get('refused',0)/n:>8.3f}"
    print(line)
