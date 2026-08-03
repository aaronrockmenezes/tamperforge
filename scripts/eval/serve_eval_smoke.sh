#!/usr/bin/env bash
# Validate the persistent-vLLM harness against KNOWN in-process numbers before trusting a
# 9-arm batch to it.
#
# serve_eval.sh has never completed a run -- it died twice on --disable-log-requests (now
# removed). The risky part is ARC and MMLU: those are LOGLIKELIHOOD tasks needing echo+logprobs
# over the completions endpoint, unlike gsm8k/humaneval/mbpp which are generative and
# straightforward. If the loglikelihood path is wrong it will not error, it will return
# PLAUSIBLE BUT WRONG numbers -- which is why this compares against a known arm rather than
# just checking the run completed.
#
# Reference (arq_clean, measured in-process 2026-08-02):
#   advbench harm 0.0000   ARC 0.3660   MMLU-12 0.5002   GSM8K 0.4496
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a

say () { echo "[$(date -u +%H:%M:%S)] $*"; }

for S in vere artfin arqher arlher; do
  if tmux ls -F "#{session_name}" 2>/dev/null | grep -qx "$S"; then
    say "[wait] $S running (vLLM cannot share the GPU with a trainer)..."
    while tmux ls -F "#{session_name}" 2>/dev/null | grep -qx "$S"; do sleep 60; done
    say "[wait] $S done, settling 30s"; sleep 30
  fi
done

TAG=smoke_arq_clean
MD=outputs/arq_clean
[ -f "$MD/model.safetensors" ] || { say "[FAIL] $MD missing -- cannot validate"; exit 1; }

# fresh slate so guards do not skip
rm -rf results/${TAG} results/${TAG}_judged results/${TAG}_arc results/${TAG}_mmlu \
       results/${TAG}_gsm8k results/${TAG}_humaneval results/${TAG}_mbpp \
       results/${TAG}_xssafe results/${TAG}_xsunsafe 2>/dev/null

say "=== smoke: serve_eval on $MD ==="
START=$(date +%s)
bash scripts/eval/serve_eval.sh "$TAG" "$MD" off
ELAPSED=$(( $(date +%s) - START ))
say "  serve_eval wall clock: $((ELAPSED/60))m $((ELAPSED%60))s"

python - "$TAG" "$ELAPSED" <<'PY'
import json, glob, os, sys
tag, elapsed = sys.argv[1], int(sys.argv[2])
REF = {"adv_harm": 0.0000, "arc": 0.3660, "mmlu": 0.5002, "gsm8k": 0.4496}
TOL = 0.02   # loglikelihood over an API should reproduce in-process closely

def lm(stem, agg=False):
    fs = sorted(glob.glob("results/%s/*/results_*.json" % stem))
    if not fs: return None
    j = json.load(open(fs[-1])); r = j["results"]
    if agg:
        ns = j["n-samples"]; ks = [k for k in r if k.startswith("mmlu")]
        if not ks: return None
        tot = sum(ns[k]["effective"] for k in ks)
        return round(sum(r[k]["acc,none"]*ns[k]["effective"] for k in ks)/tot, 4)
    v = list(r.values())[0]
    for m in ("exact_match,strict-match","acc_norm,none","pass@1,create_test","pass_at_1,none"):
        if m in v: return round(v[m], 4)
    return None

def saf(t):
    p = "results/%s_judged/summary.json" % t
    if not os.path.exists(p): return None
    s = json.load(open(p)); return round(s["usefulness_counts"].get("harmful_actionable",0)/s["n"], 4)

got = {"adv_harm": saf(tag), "arc": lm(tag+"_arc"),
       "mmlu": lm(tag+"_mmlu", agg=True), "gsm8k": lm(tag+"_gsm8k")}
print("\n%-10s %10s %10s %8s  %s" % ("metric", "in-process", "via-API", "delta", "verdict"))
print("-"*56)
bad = []
for k, ref in REF.items():
    g = got[k]
    if g is None:
        print("%-10s %10.4f %10s %8s  MISSING" % (k, ref, "-", "-")); bad.append(k); continue
    d = g - ref
    ok = abs(d) <= TOL
    print("%-10s %10.4f %10.4f %+8.4f  %s" % (k, ref, g, d, "ok" if ok else "MISMATCH"))
    if not ok: bad.append(k)
for extra in ("humaneval", "mbpp"):
    v = lm(tag+"_"+extra)
    print("%-10s %10s %10s %8s  (no in-process ref)" % (extra, "-", ("%.4f"%v) if v else "-", "-"))
print("\nwall clock: %dm%02ds  (in-process baseline was ~20m/arm)" % (elapsed//60, elapsed%60))
print("VERDICT:", "PASS -- harness is trustworthy for the batch" if not bad
      else "FAIL on %s -- use in-process for those tasks" % ",".join(bad))
PY
say "=== smoke DONE ==="
