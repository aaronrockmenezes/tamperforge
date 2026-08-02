import json, os
rows = [
 ("THINKING-OFF (the trained regime)", None, None),
 ("  v8 clean",                 "results/xcm_v8_clean_think_judged", "SKIP"),
 ("  version_A clean",          "results/xcm_va_clean_off_judged", None),
 ("  version_A + rank-1",       "results/xoff_va_rank1_judged", None),
 ("  version_A + surgical k16", "results/xoff_va_surg_k16_judged", None),
 ("  v8 + surgical k16",        "results/xoff_v8_surg_k16_judged", None),
 ("", None, None),
 ("THINKING-ON", None, None),
 ("  v8 clean",                 "results/xcm_v8_clean_think_judged", None),
 ("  version_A clean",          "results/xva_clean_judged", None),
 ("  version_A + rank-1",       "results/xva_rank1_judged", None),
 ("  version_A + surgical k16", "results/xva_surg_k16_judged", None),
 ("  v8 + surgical k16",        "results/xtest_v8_surg_k16_judged", None),
]
hdr = f"{'':<30}{'harmful':>10}{'gibber':>9}{'refused':>9}"
print(hdr); print("-"*len(hdr))
for name, d, skip in rows:
    if d is None: print(name); continue
    if skip: continue
    p = f"{d}/summary.json"
    if not os.path.exists(p): print(f"{name:<30}{'(missing)':>10}"); continue
    s = json.load(open(p)); u = s["usefulness_counts"]; n = s["n"]
    print(f"{name:<30}{u.get('harmful_actionable',0)/n:>10.4f}"
          f"{u.get('gibberish',0)/n:>9.4f}{u.get('refused',0)/n:>9.4f}")
