import json, glob, os
arms=[("xcm_v8_clean","v8 clean"),("xtest_v8_surg_k16","v8 + surgical"),
      ("xva_clean","A clean"),("xva_rank1","A + rank-1"),("xva_surg_k16","A + surgical"),
      ("xvb_clean","B clean"),("xvb_rank1","B + rank-1"),("xvb_surg_k16","B + surgical"),
      ("heretic_va_s500_t85","A + heretic t85"),("heretic_va_s500_t175","A + heretic t175")]
print(f"{'arm':<22}{'inst_strict':>12}{'prompt_strict':>14}")
print("-"*48)
for tag,lbl in arms:
    f=glob.glob(f"results/ifev_{tag}/**/results_*.json",recursive=True)
    if not f: print(f"{lbl:<22}{'pending':>12}"); continue
    d=json.load(open(f[0])).get("results",{}).get("ifeval",{})
    i=[v for k,v in d.items() if k.startswith("inst_level_strict_acc")]
    p=[v for k,v in d.items() if k.startswith("prompt_level_strict_acc")]
    print(f"{lbl:<22}{(i[0] if i else float('nan')):>12.4f}{(p[0] if p else float('nan')):>14.4f}")
