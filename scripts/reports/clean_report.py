import json, os
rows = [("v8 clean              thinking-ON",  "results/xcm_v8_clean_think_judged"),
        ("version_A s500 clean  thinking-ON",  "results/xva_clean_judged"),
        ("version_A s500 clean  thinking-OFF", "results/xcm_va_clean_off_judged")]
hdr = f"{'model / mode':<38}{'n':>5}{'harmful':>10}{'gibber':>9}{'refused':>9}"
print(hdr); print("-"*len(hdr))
for name, d in rows:
    p = f"{d}/summary.json"
    if not os.path.exists(p): print(f"{name:<38}{'(missing)':>5}"); continue
    s = json.load(open(p)); u = s["usefulness_counts"]; n = s["n"]
    print(f"{name:<38}{n:>5}{u.get('harmful_actionable',0)/n:>10.4f}"
          f"{u.get('gibberish',0)/n:>9.4f}{u.get('refused',0)/n:>9.4f}")
print("\nreference: v8 clean thinking-OFF (campaign) = 0.0115 harmful / 0.9827 refused")
