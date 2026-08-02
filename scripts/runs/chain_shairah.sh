#!/usr/bin/env bash
# Overnight chain: Shairah extended-refusal baseline through the FULL protocol.
#
#   Stage 1  materialise + evaluate shairah clean / rank-1 / surgical, both architectures
#   Stage 2  heretic 200 trials x 3 seeds per architecture, then evaluate the winners
#
# Every arm gets: AdvBench-520 (judged), ARC, MMLU-12, GSM8K, HumanEval, MBPP,
# XSTest safe + unsafe. XSTest matters most here -- 2026-08-02 found every version of OUR
# defense fails ~2/3 of safe-but-scary prompts via lexically-triggered degeneration, and
# extended-refusal tuning has no gibberish objective, so this is where it should win.
#
# HARD RULES encoded here (all learned the expensive way, see CLAUDE.md):
#   * Guard on the ARTIFACT, never the directory -- an empty dir from a killed job made
#     four eval arms silently vanish on 2026-08-01.
#   * Never `pkill -f` a global pattern. Scope to our own child via `pgrep -P`.
#   * NEVER run vLLM while a trainer holds the GPU: vLLM's startup memory profile ASSERTS
#     if free VRAM changes during profiling, which is not an OOM and no util setting avoids.
#     It also orphans an EngineCore holding ~7GB (PPID 1) that must be kill -9'd.
#   * Heretic studies are ~3.7GB and safe 3-up; vLLM stages are NOT. Studies parallel,
#     evals serial.
#   * Heretic winners are REPLAYED, not materialised interactively: validated 2026-08-02,
#     replay matches the real checkpoint to the exact count on harmful_actionable (34/520)
#     and within stderr on GSM8K. heretic has no --save flag; its save menu is interactive.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
export HF_ALLOW_CODE_EVAL=1

LOG=logs/eval/chain_shairah_$(date -u +%Y%m%dT%H%M%S).log
mkdir -p logs/eval outputs results
UTIL=0.45
PDIR=scripts/external_benches/prompts
MMLU12=mmlu_abstract_algebra,mmlu_business_ethics,mmlu_college_computer_science,mmlu_computer_security,mmlu_econometrics,mmlu_high_school_biology,mmlu_high_school_us_history,mmlu_machine_learning,mmlu_philosophy,mmlu_professional_medicine,mmlu_sociology,mmlu_world_religions

say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== chain_shairah START ==="

# Do not share the GPU with the trainer. Wait for it, whatever else happens.
if tmux has-session -t shairah 2>/dev/null; then
  say "[wait] shairah training running..."
  while tmux has-session -t shairah 2>/dev/null; do sleep 60; done
  say "[wait] training finished, settling 30s"; sleep 30
fi
if tmux has-session -t xsvavc 2>/dev/null; then
  say "[wait] xsvavc matrix running..."
  while tmux has-session -t xsvavc 2>/dev/null; do sleep 60; done
  say "[wait] xsvavc finished, settling 30s"; sleep 30
fi

have () { find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; }

# Reap any orphaned EngineCore (PPID 1) before starting a vLLM stage: one of these
# silently held 6.7GB for 6 minutes on 2026-08-02.
reap_orphans () {
  for p in $(pgrep -f 'VLLM::EngineCore' 2>/dev/null); do
    if [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" = "1" ]; then
      say "  [reap] orphaned EngineCore $p"; kill -9 "$p" 2>/dev/null || true
    fi
  done
}

run_lm () {   # $1=outdir $2=model $3=tasks $4=fewshot
  local out="$1" md="$2" tasks="$3" shots="$4"
  have "$out" && { say "  [skip] $out"; return 0; }
  reap_orphans
  say "  lm_eval $out"
  lm_eval --model vllm \
    --model_args "pretrained=${md},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=${UTIL}" \
    --tasks "$tasks" --num_fewshot "$shots" --batch_size auto \
    --confirm_run_unsafe_code --output_path "$out" >>"$LOG" 2>&1 &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if have "$out"; then
      sleep 5
      local ep; ep="$(pgrep -P "$pid" -f 'VLLM::EngineCore' | head -1 || true)"
      [ -n "$ep" ] && kill -TERM "$ep" 2>/dev/null || true
      sleep 10; kill -TERM "$pid" 2>/dev/null || true; break
    fi
    sleep 10; waited=$((waited+10))
    [ "$waited" -gt 2400 ] && { say "  [TIMEOUT] $out"; kill -TERM "$pid" 2>/dev/null; break; }
  done
  wait "$pid" 2>/dev/null || true
  sleep 5
  have "$out" || say "  [FAIL] $out"
}

gen_and_judge () {   # $1=tag $2=model $3=promptfile|advbench $4=thinking
  local tag="$1" md="$2" src="$3" think="$4"
  if [ -f "results/${tag}/generations.jsonl" ]; then
    say "  [skip] gen $tag"
  else
    reap_orphans
    say "  gen $tag"
    if [ "$src" = "advbench" ]; then
      python -u experiments/p0_baseline_eval.py --run-id "$tag" --model-id "$md" \
        --prompt-source advbench --advbench-source walledai --advbench-split train \
        --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 --max-new-tokens 512 --max-length 4096 \
        --qwen-thinking "$think" --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
        --vllm-gpu-memory-utilization "$UTIL" --vllm-temperature 0.0 --vllm-top-p 1.0 >>"$LOG" 2>&1
    else
      python -u experiments/p0_baseline_eval.py --run-id "$tag" --model-id "$md" \
        --prompt-file "$src" \
        --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 --max-new-tokens 512 --max-length 4096 \
        --qwen-thinking "$think" --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
        --vllm-gpu-memory-utilization "$UTIL" --vllm-temperature 0.0 --vllm-top-p 1.0 >>"$LOG" 2>&1
    fi
    [ -f "results/${tag}/generations.jsonl" ] || { say "  [FAIL] gen $tag"; return 0; }
  fi
  [ -f "results/${tag}_judged/summary.json" ] && { say "  [skip] judge $tag"; return 0; }
  say "  judge $tag"
  python -u experiments/judge_generations.py --generations "results/${tag}/generations.jsonl" \
    --run-id "${tag}_judged" --num-workers 32 >>"$LOG" 2>&1
  [ -f "results/${tag}_judged/summary.json" ] || say "  [FAIL] judge $tag"
}

eval_all () {   # $1=tag $2=modeldir $3=thinking
  local tag="$1" md="$2" think="$3"
  say "##### eval_all $tag #####"
  [ -f "$md/model.safetensors" ] || { say "  [MISSING] $md"; return 0; }
  gen_and_judge "$tag" "$md" advbench "$think"
  gen_and_judge "${tag}_xssafe"   "$md" "$PDIR/xstest_safe.jsonl"   "$think"
  gen_and_judge "${tag}_xsunsafe" "$md" "$PDIR/xstest_unsafe.jsonl" "$think"
  run_lm "results/${tag}_arc"       "$md" arc_challenge 0
  run_lm "results/${tag}_mmlu"      "$md" "$MMLU12"     5
  run_lm "results/${tag}_gsm8k"     "$md" gsm8k         5
  run_lm "results/${tag}_humaneval" "$md" humaneval     0
  run_lm "results/${tag}_mbpp"      "$md" mbpp          3
}

# Attacked weights are reproducible from the checkpoint + trial json, and the box has
# ~34GB free against ~30GB of would-be model dirs. Drop them once their results exist.
# NEVER drop *_clean -- stage 2's heretic studies read it.
drop_weights () {   # $1=modeldir $2=tag
  case "$1" in *_clean) return 0;; esac
  if [ -f "results/${2}_judged/summary.json" ] && have "results/${2}_gsm8k"; then
    say "  [cleanup] $1 ($(du -sh "$1" 2>/dev/null | cut -f1))"
    rm -rf "$1"
  else
    say "  [cleanup-skip] $1 -- results incomplete, keeping weights"
  fi
}

# ============================ STAGE 1: attacks + evals ============================
# arch: tag_prefix | checkpoint | model-id | direction-layer | thinking
for spec in \
  "shq|outputs/shairah_qwen_500.pt|Qwen/Qwen3-0.6B|20|off" \
  "shl|outputs/shairah_llama_500.pt|meta-llama/Llama-3.2-1B-Instruct|13|default"
do
  IFS='|' read -r P CK MID DL TH <<<"$spec"
  if [ ! -f "$CK" ]; then say "[MISSING] $CK -- skipping $P"; continue; fi
  say "=== STAGE 1: $P (DL=$DL) ==="

  [ -f "outputs/${P}_clean/model.safetensors" ] || python -u experiments/save_p1b_checkpoint.py \
    --checkpoint "$CK" --model-id "$MID" --attack none --out "outputs/${P}_clean" >>"$LOG" 2>&1
  eval_all "${P}_clean" "outputs/${P}_clean" "$TH"

  [ -f "outputs/${P}_rank1/model.safetensors" ] || python -u experiments/v11_surgical_ablation.py \
    --model-id "$MID" --checkpoint "$CK" --direction-layer "$DL" --cap-rank 0 \
    --out "outputs/${P}_rank1" >>"$LOG" 2>&1
  eval_all "${P}_rank1" "outputs/${P}_rank1" "$TH"
  drop_weights "outputs/${P}_rank1" "${P}_rank1"

  [ -f "outputs/${P}_surg_k16/model.safetensors" ] || python -u experiments/v11_surgical_ablation.py \
    --model-id "$MID" --checkpoint "$CK" --direction-layer "$DL" --cap-rank 16 \
    --out "outputs/${P}_surg_k16" >>"$LOG" 2>&1
  eval_all "${P}_surg_k16" "outputs/${P}_surg_k16" "$TH"
  drop_weights "outputs/${P}_surg_k16" "${P}_surg_k16"
done

# ============================ STAGE 2: heretic x3 seeds ============================
for spec in \
  "shq|Qwen/Qwen3-0.6B|off|outputs/shairah_qwen_500.pt" \
  "shl|meta-llama/Llama-3.2-1B-Instruct|default|outputs/shairah_llama_500.pt"
do
  IFS='|' read -r P MID TH CK <<<"$spec"
  [ -f "outputs/${P}_clean/model.safetensors" ] || { say "[skip] no ${P}_clean, no heretic"; continue; }
  say "=== STAGE 2: heretic x3 on $P ==="

  # studies run 3-up: ~3.7GB each, measured safe. No vLLM concurrently.
  pids=()
  for S in 0 1 2; do
    HLOG="logs/heretic/heretic_${P}_s${S}.log"
    mkdir -p logs/heretic
    if grep -aq "Running trial 200 of" "$HLOG" 2>/dev/null; then say "  [skip] study ${P} s${S}"; continue; fi
    rm -rf "/tmp/hcp_${P}_s${S}"
    say "  study ${P} s${S} launching"
    heretic --model "outputs/${P}_clean" --n-trials 200 --seed "$S" \
      --study-checkpoint-dir "/tmp/hcp_${P}_s${S}" < /dev/null > "$HLOG" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" 2>/dev/null || true; done
  say "  studies done for $P"

  for S in 0 1 2; do
    HLOG="logs/heretic/heretic_${P}_s${S}.log"
    TAG="${P}_her_s${S}"
    [ -f "$HLOG" ] || continue
    if [ ! -f "results/${TAG}_trial.json" ]; then
      python - "$HLOG" "$TAG" <<'PY' >>"$LOG" 2>&1
import json, sys
sys.path.insert(0, "experiments")
from version_c_loop import parse_trials, pick_winners
t = parse_trials(open(sys.argv[1], encoding="utf-8", errors="replace").read())
w = pick_winners(t, k=1, kl_max=0.5)
if w:
    json.dump({"heretic_trials": {"t%d" % w[0]["trial"]: w[0]}},
              open("results/%s_trial.json" % sys.argv[2], "w"), indent=2)
    print("winner trial %d ref %d kl %.4f" % (w[0]["trial"], w[0]["refusals"], w[0]["kl"]))
else:
    print("NO WINNER")
PY
    fi
    [ -f "results/${TAG}_trial.json" ] || { say "  [FAIL] no winner ${TAG}"; continue; }
    T=$(python -c "import json;print(list(json.load(open('results/${TAG}_trial.json'))['heretic_trials'])[0])" 2>/dev/null)
    [ -z "$T" ] && { say "  [FAIL] unparseable winner ${TAG}"; continue; }
    say "  ${TAG} winner ${T}"
    D="outputs/${TAG}_att"
    [ -f "$D/model.safetensors" ] || python -u experiments/version_c_replay.py \
      --model-id "$MID" --checkpoint "$CK" --trial "$T" \
      --params-json "results/${TAG}_trial.json" \
      --direction-recipe heretic --application heretic_full --out "$D" >>"$LOG" 2>&1
    eval_all "$TAG" "$D" "$TH"
    drop_weights "$D" "$TAG"
  done
done

say "=== chain_shairah DONE ==="
say "disk: $(df -h /workspace | tail -1)"
