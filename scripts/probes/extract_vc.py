import json, sys
sys.path.insert(0, "experiments")
from version_c_loop import parse_trials
t = parse_trials(open("logs/training_runs/heretic_version_c.log",
                      encoding="utf-8", errors="replace").read())
want = {71, 156, 47}
sel = {"t%d" % x["trial"]: x for x in t if x["trial"] in want}
json.dump({"heretic_trials": sel}, open("results/heretic_vc_trials.json", "w"), indent=2)
for k, v in sorted(sel.items()):
    print(k, "ref", v["refusals"], "kl", v["kl"], "dir", v["direction_index"])
