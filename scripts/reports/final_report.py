import json, sys, os
rows = [("version_A s500 clean",      "results/xva_clean_judged"),
        ("version_A s500 rank-1",     "results/xva_rank1_judged"),
        ("version_A s500 surgical k16","results/xva_surg_k16_judged"),
        ("v8 surgical k16 (control)",  "results/xtest_v8_surg_k16_judged")]
hdr = f"{'model / attack':<30}{'n':>5}{'harmful':>10}{'gibber':>9}{'refused':>9}{'ASR':>8}"
print(hdr); print("-"*len(hdr))
for name, d in rows:
    if not os.path.exists(f"{d}/summary.json"):
        print(f"{name:<30}{'(missing)':>5}"); continue
    s = json.load(open(f"{d}/summary.json")); u = s["usefulness_counts"]; n = s["n"]
    print(f"{name:<30}{n:>5}{u.get('harmful_actionable',0)/n:>10.4f}"
          f"{u.get('gibberish',0)/n:>9.4f}{u.get('refused',0)/n:>9.4f}{s['judge_asr']:>8.4f}")
print()
print("campaign reference  v8 surgical k16 = 0.4481 harmful / 0.1615 gib / 0.3673 ref")
print("campaign reference  v8 clean        = 0.0115 harmful / 0.9827 refused")
print("campaign reference  v8 rank-1       = 0.0000 harmful / 0.9442 gibberish")
