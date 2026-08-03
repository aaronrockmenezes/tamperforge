#!/usr/bin/env bash
# Can plain benign SFT give version_B its voice back without giving up the wall?
#
# THE QUESTION. version_B is the only model that passes gate 2 (heretic harm 0.3212) and it
# fails gate 1 on conversational quality alone (MT-Bench 3.33 vs base 4.54; heretic'd 2.96).
# Its benchmark capability is fine -- GSM8K 0.3889 clean, 95% of that under heretic -- so what
# is broken is FORM, not knowledge. If a few hundred assistant-style rows restore the form and
# the wall survives, version_F does not need to be trained at all: the recipe becomes
# "version_B, then repair".
#
# WHY SELF-DISTILLATION FROM BASE, not an off-the-shelf SFT set. The target is specifically
# base Qwen's response STYLE on open-ended prompts, which is what MT-Bench grades and what
# version_B lost. Alpaca's own gold responses are terse and would not teach the long-form shape.
# So: base generates the responses, version_B trains on them. Nothing new is taught, only form.
#
# PROMPTS ARE ALPACA, NOT MT-BENCH. Training on MT-Bench prompts and then scoring MT-Bench
# would be training on test. Alpaca no-input instructions are open-ended and disjoint from it.
#
# WHAT DECIDES IT. Three numbers, in order, and the third is the one that can kill it:
#   1. MT-Bench    -- did the voice come back? (vb_clean 3.33, base 4.54, gate-1 bar 4.04)
#   2. AdvBench    -- is the clean model still safe? (vs vb_clean's own harm)
#   3. rank-1      -- does the POISON PILL survive? version_B under rank-1 craters GSM8K to
#                     0.0091 (2%). If SFT restores fluency but rank-1 stops cratering, the
#                     repair dissolved the entanglement and the whole result with it.
# A pass on 1+2 with a fail on 3 is the worst outcome and the most likely one -- benign SFT is
# a mild fine-tuning attack, and FTR is a closed thread precisely because FT beats us.
#
# ONE GPU. vLLM cannot share it with a trainer, so this serves, tears the server down, trains,
# then serves again. Never run it alongside a training job.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/eval logs/probes results outputs

N_ROWS="${N_ROWS:-1000}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
CKPT=outputs/version_b_qwen_500.pt
BASE_HF=outputs/xbase_clean_hf
PROMPTS=scripts/external_benches/prompts/alpaca_sft_${N_ROWS}.jsonl
DEMOS=results/sft_demos_base/generations.jsonl
OUT=outputs/vb_sft${N_ROWS}
PORT=8765

say () { echo "[$(date -u +%H:%M:%S)] $*"; }
[ -f "$CKPT" ] || { say "[FAIL] missing $CKPT"; exit 1; }
[ -f "$BASE_HF/model.safetensors" ] || { say "[FAIL] missing $BASE_HF"; exit 1; }

# ---- 0. prompts: alpaca no-input instructions, disjoint from MT-Bench -----------------
if [ -f "$PROMPTS" ]; then
  say "[skip] $PROMPTS exists ($(wc -l < "$PROMPTS") rows)"
else
  say "=== building $N_ROWS alpaca prompts ==="
  python - "$PROMPTS" "$N_ROWS" <<'PY'
import json, random, sys
from datasets import load_dataset
out, n = sys.argv[1], int(sys.argv[2])
ds = [r for r in load_dataset("tatsu-lab/alpaca", split="train")
      if not r["input"].strip() and 20 <= len(r["instruction"]) <= 300]
random.Random(42).shuffle(ds)
rows = ds[:n]
assert len(rows) == n, f"only {len(rows)} usable alpaca rows, wanted {n}"
mt = {json.loads(l)["prompt"].strip()
      for l in open("scripts/external_benches/prompts/mtbench_t1.jsonl") if l.strip()}
rows = [r for r in rows if r["instruction"].strip() not in mt]
assert len(rows) == n, "alpaca overlapped MT-Bench -- would be training on test"
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps({"prompt": r["instruction"].strip()}) + "\n")
print(f"wrote {len(rows)} -> {out}")
PY
  [ -f "$PROMPTS" ] || { say "[FAIL] prompt build"; exit 1; }
fi

# ---- server helpers (one at a time; the trainer needs the whole GPU) ------------------
SP=""
serve () {   # $1=tag  $2=model-dir
  for p in $(pgrep -f 'VLLM::EngineCore|vllm serve' 2>/dev/null); do
    [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" = "1" ] && kill -9 "$p" 2>/dev/null
  done
  sleep 3
  ss -tln 2>/dev/null | grep -q ":${PORT} " && { say "  [FAIL] port ${PORT} busy"; return 1; }
  vllm serve "$2" --served-model-name "$1" --port "$PORT" \
    --gpu-memory-utilization 0.85 --max-model-len 4096 --dtype bfloat16 \
    > "logs/eval/vllm_${1}.log" 2>&1 &
  SP=$!
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"$1\"" && return 0
    kill -0 "$SP" 2>/dev/null || { say "  [FAIL] server died"; tail -15 "logs/eval/vllm_${1}.log"; return 1; }
    sleep 5
  done
  say "  [FAIL] server never advertised $1"; return 1
}
unserve () {
  [ -n "$SP" ] || return 0
  for c in $(pgrep -P "$SP" 2>/dev/null); do kill -TERM "$c" 2>/dev/null; done
  kill -TERM "$SP" 2>/dev/null; sleep 8; kill -9 "$SP" 2>/dev/null; SP=""; sleep 5
}
trap unserve EXIT

# ---- 1. base generates the demonstrations --------------------------------------------
if [ -f "$DEMOS" ]; then
  say "[skip] demos exist ($(wc -l < "$DEMOS") rows)"
else
  say "=== base generating $N_ROWS demos ==="
  serve sftbase "$BASE_HF" || exit 1
  python -u experiments/gen_via_api.py --run-id sft_demos_base --served-model sftbase \
    --base-url "http://127.0.0.1:${PORT}/v1" --prompt-file "$PROMPTS" \
    --qwen-thinking off --max-new-tokens 768 --num-workers 32 2>&1 | tail -2
  unserve
  [ -f "$DEMOS" ] || { say "[FAIL] no demos"; exit 1; }
fi
say "  demos: $(wc -l < "$DEMOS") rows, $(python3 -c "
import json;rs=[json.loads(l) for l in open('$DEMOS')]
print(sum(1 for r in rs if not (r.get('response') or '').strip()),'empty')")"

# ---- 2. repair version_B on them ------------------------------------------------------
if [ -f "$OUT/model.safetensors" ]; then
  say "[skip] $OUT exists"
else
  say "=== SFT version_B on $N_ROWS rows, $EPOCHS epoch(s), lr $LR ==="
  python -u experiments/ft_attack.py --model-id Qwen/Qwen3-0.6B \
    --checkpoint "$CKPT" --demos "$DEMOS" --n-shots "$N_ROWS" \
    --ft-epochs "$EPOCHS" --ft-lr "$LR" --max-len 1024 --out "$OUT" \
    2>&1 | tee "logs/training_runs/vb_sft${N_ROWS}.log" | grep -aE "^\[ft\]"
  [ -f "$OUT/model.safetensors" ] || { say "[FAIL] SFT produced nothing"; exit 1; }
fi

say "  --- smoke5 gate ---"
python scripts/probes/smoke5.py "$OUT" --max-new 120 --modes nothink,default \
  2>&1 | tee "logs/probes/smoke5_vb_sft${N_ROWS}.log" | grep -aE "^\[|^MODEL|^====" | head -30

# ---- 3. rank-1 ablation of the repaired model (does the poison pill survive?) ---------
ABL="${OUT}_rank1"
if [ -f "$ABL/model.safetensors" ]; then say "[skip] $ABL exists"; else
  say "=== rank-1 ablation of repaired model ==="
  # --model-id takes a local dir (from_pretrained), so the repaired HF dir IS the model;
  # no --checkpoint, the weights are already baked in.
  python -u experiments/v11_surgical_ablation.py --model-id "$OUT" \
    --direction-layer 20 --cap-rank 0 --out "$ABL" 2>&1 | tail -3
fi

# ---- 4. the three numbers -------------------------------------------------------------
gen_and_judge () {   # $1=tag $2=model-dir $3...=prompt args
  local tag="$1" md="$2"; shift 2
  [ -f "$md/model.safetensors" ] || { say "  [MISSING] $md"; return 0; }
  if [ -f "results/${tag}/generations.jsonl" ]; then say "  [skip] gen $tag"; else
    serve "$tag" "$md" || return 0
    python -u experiments/gen_via_api.py --run-id "$tag" --served-model "$tag" \
      --base-url "http://127.0.0.1:${PORT}/v1" --qwen-thinking off \
      --num-workers 32 "$@" 2>&1 | tail -2
    unserve
  fi
  case "$tag" in *_mtb) return 0 ;; esac
  [ -f "results/${tag}_judged/summary.json" ] && { say "  [skip] judge $tag"; return 0; }
  python -u experiments/judge_generations.py --generations "results/${tag}/generations.jsonl" \
    --run-id "${tag}_judged" --num-workers 32 2>&1 | tail -2
}

say "=== MT-Bench + AdvBench on repaired model, AdvBench on its rank-1 ==="
gen_and_judge "mtb_vb_sft"    "$OUT" --prompt-file scripts/external_benches/prompts/mtbench_t1.jsonl --max-new-tokens 768
gen_and_judge "vbsft_adv"     "$OUT" --prompt-source advbench
gen_and_judge "vbsft_r1_adv"  "$ABL" --prompt-source advbench

say "=== absolute MT-Bench (pinned judge) ==="
python -u experiments/mtbench_single.py --tags mtb_xbase_clean mtb_vb_clean mtb_vb_heretic mtb_vb_sft \
  2>&1 | grep -avE "it/s\]|\r"

say "=== GSM8K: repaired clean vs its rank-1 (the poison pill check) ==="
for spec in "vbsft_clean:$OUT" "vbsft_rank1:$ABL"; do
  nm="${spec%%:*}"; md="${spec#*:}"
  [ -f "$md/model.safetensors" ] || continue
  find "results/${nm}_gsm8k" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . \
    && { say "  [skip] ${nm}_gsm8k"; continue; }
  serve "$nm" "$md" || continue
  lm_eval --model local-completions \
    --model_args "model=${nm},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,tokenizer=${md}" \
    --tasks gsm8k --num_fewshot 5 --batch_size 1 --output_path "results/${nm}_gsm8k" 2>&1 | tail -3
  unserve
done

say "=== VB SFT REPAIR DONE ==="
df -h /workspace | tail -1
