#!/usr/bin/env bash
# Training-only Version G fleet for a 96GB RTX PRO 6000.
# No DL sweep, benchmark generation, API judging, Heretic, or lm-eval runs here.
# Each model: one-step architecture smoke -> 500-step train -> clean HF materialization
# -> private Hugging Face upload -> remote artifact verification.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

PY="${PY:-python}"
[ -x /venv/main/bin/python ] && PY=/venv/main/bin/python
GPU="${GPU:-0}"
STEPS="${STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-100}"
HF_REPO="${HF_REPO:-aaronrockmenezes/tamperforge}"
HF_PREFIX="${HF_PREFIX:-version_g_final}"
RANK_K_ESTIMATOR="${RANK_K_ESTIMATOR:-arditi_residual}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
ONLY_MODELS=",${ONLY_MODELS:-},"       # comma-separated slugs; empty = all
SMOKE="${SMOKE:-1}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOG_ROOT="logs/training_runs/pro6000"
mkdir -p "$LOG_ROOT" outputs

# Ministral's requested Instruct repo is FP8. Full-weight training uses its official
# BF16-equivalent repo; training float8 parameters with AdamW is unsupported.
MODELS=(
  "microsoft/Phi-4-mini-instruct|phi4mini|13"
  "mistralai/Ministral-3-3B-Instruct-2512-BF16|ministral3_3b|15"
  "meta-llama/Llama-3.2-3B-Instruct|llama32_3b|14"
  "Qwen/Qwen3-4B-Instruct-2507|qwen3_4b_2507|19"
  "google/gemma-4-E2B-it|gemma4_e2b|17"
  "Qwen/Qwen3.5-4B|qwen35_4b|16"
)

say() { echo "[$(date -u +%FT%TZ)] $*"; }
die() { say "[FAIL] $*"; exit 1; }
selected() { [ "$ONLY_MODELS" = ",," ] || [[ "$ONLY_MODELS" == *",$1,"* ]]; }

case "$GPU:$STEPS:$SAVE_EVERY" in
  *[!0-9:]*|:*|*::*|*:) die "GPU/STEPS/SAVE_EVERY must be nonnegative integers";;
esac
case "$SMOKE:$GRAD_CHECKPOINT:$DRY_RUN" in
  0:0:0|0:0:1|0:1:0|0:1:1|1:0:0|1:0:1|1:1:0|1:1:1) ;;
  *) die "SMOKE, GRAD_CHECKPOINT, and DRY_RUN must be 0 or 1";;
esac
[ "$STEPS" -gt 0 ] || die "STEPS must be positive"
case "$RANK_K_ESTIMATOR" in svd|arditi_residual|partitioned) ;; *) die "bad RANK_K_ESTIMATOR";; esac
for f in experiments/train_version_g_final.py experiments/save_p1b_checkpoint.py \
         data/harm_targets_qwen.json data/extended_refusals_advbench.json; do
  [ -e "$f" ] || die "missing $f"
done
selected_n=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r model slug dl <<< "$spec"
  selected "$slug" || continue
  selected_n=$((selected_n + 1))
  printf '%-18s layer=%-3s %s\n' "$slug" "$dl" "$model"
done
[ "$selected_n" -gt 0 ] || die "ONLY_MODELS selected no known model slugs"
if [ "$DRY_RUN" = 1 ]; then
  say "DRY_RUN: $selected_n models; steps=$STEPS estimator=$RANK_K_ESTIMATOR; nothing launched"
  exit 0
fi
command -v hf >/dev/null || die "hf CLI not installed"
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN unset"
$PY - "$GPU" <<'PY' || die "CUDA/PRO6000 preflight"
import sys, torch, transformers
g = int(sys.argv[1])
assert torch.cuda.is_available() and g < torch.cuda.device_count(), "requested CUDA GPU absent"
p = torch.cuda.get_device_properties(g)
assert p.total_memory >= 80 * 2**30, f"expected >=80 GiB, found {p.total_memory/2**30:.1f}"
assert tuple(map(int, transformers.__version__.split(".")[:2])) >= (5, 9), transformers.__version__
print(f"gpu{g}: {p.name}, {p.total_memory/2**30:.1f} GiB; transformers={transformers.__version__}")
PY
$PY - "$HF_REPO" <<'PY' || die "private Hugging Face repo preflight"
import sys
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info(sys.argv[1])
assert info.private is True, f"{sys.argv[1]} must already exist and be private"
print(f"private HF repo verified: {sys.argv[1]}")
PY
df -h . | tail -1
export CUDA_VISIBLE_DEVICES="$GPU"

GRAD_ARGS=(--no-grad-checkpoint)
[ "$GRAD_CHECKPOINT" = 0 ] || GRAD_ARGS=(--grad-checkpoint)

train_args() {
  local model="$1" out="$2" run="$3" dl="$4" steps="$5" save_every="$6"
  shift 6
  "$PY" -u experiments/train_version_g_final.py \
    --model-id "$model" --out "$out" --run-id "$run" \
    --train-scope all --abliterate-layers all --attack-ensemble \
    --attack-profile version_g_final --attack-layers all --direction-layer "$dl" \
    --vg-rank-k-prob 0.10 --version-g-attack-ranks 1,2,4,8,16 \
    --version-g-rank-estimator "$RANK_K_ESTIMATOR" \
    --vg-surgical-prob 0.40 --vg-surgical-cap-ranks 2,4,8,16 --vg-heretic-prob 0.35 \
    "${GRAD_ARGS[@]}" --recompute-direction-every 25 \
    --lambda-rr 4 --rr-center --harm-targets data/harm_targets_qwen.json --rr-layers last_half \
    --lambda-gib 0 --stage2-lambda-gib 0 --gib-mode argmax \
    --lambda-uncensor 4 --uncensor-margin 4 --lambda-harm 4 --harm-margin 4 \
    --lambda-safe 4 --stage2-lambda-safe 4 --lambda-reg 0.1 --lambda-clean 3 \
    --clean-gen-prompts 2 --clean-gen-tokens 64 --clean-start-step 0 --clean-ramp-steps 100 \
    --refusal-file data/extended_refusals_advbench.json --refusal-max-len 384 \
    --advbench-preview-tokens 0 --gsm8k-probe-n 0 --advbench-judge-n 0 \
    --n-direction 256 --vg-n-capability 256 \
    --steps "$steps" --eval-every 0 --save-every "$save_every" --lr 1e-5 --seed 42 \
    --qwen-thinking off "$@"
}

for spec in "${MODELS[@]}"; do
  IFS='|' read -r model slug dl <<< "$spec"
  selected "$slug" || continue
  run="version_g_${slug}_rrcenter"
  ck="outputs/${run}.pt"
  clean="outputs/${run}_clean"
  log="$LOG_ROOT/${run}.log"
  remote="$HF_PREFIX/$slug"
  say "=== $slug :: $model ==="

  if [ "$SMOKE" = 1 ] && [ ! -s "$LOG_ROOT/${run}.smoke.ok" ]; then
    smoke_dir=$(mktemp -d "/tmp/tamperforge_${slug}_smoke.XXXXXX")
    smoke_ck="$smoke_dir/smoke.pt"
    say "one-step architecture/gradient smoke"
    train_args "$model" "$smoke_ck" "${run}_smoke" "$dl" 1 0 --smoke \
      2>&1 | tee "$LOG_ROOT/${run}.smoke.log"
    [ -s "$smoke_ck" ] || die "$slug smoke produced no checkpoint"
    : > "$LOG_ROOT/${run}.smoke.ok"
  fi

  if [ ! -s "$ck" ]; then
    say "training $STEPS steps; no intermediate evaluation; checkpoint every $SAVE_EVERY"
    train_args "$model" "$ck" "$run" "$dl" "$STEPS" "$SAVE_EVERY" \
      2>&1 | tee -a "$log"
  fi
  [ -s "$ck" ] || die "$slug training produced no checkpoint"

  if [ ! -s "$clean/config.json" ]; then
    say "materializing clean HF model"
    "$PY" -u experiments/save_p1b_checkpoint.py --model-id "$model" \
      --checkpoint "$ck" --attack none --out "$clean" 2>&1 | tee -a "$log"
  fi
  [ -s "$clean/config.json" ] || die "$slug clean materialization failed"
  if [ ! -s "$log" ]; then
    say "checkpoint/model reused; creating upload audit log"
    printf '[%s] reused existing checkpoint=%s clean=%s\n' \
      "$(date -u +%FT%TZ)" "$ck" "$clean" > "$log"
  fi

  manifest="$LOG_ROOT/${run}.manifest.json"
  "$PY" - "$model" "$slug" "$ck" "$clean" "$RANK_K_ESTIMATOR" "$STEPS" > "$manifest" <<'PY'
import hashlib, json, pathlib, subprocess, sys
model, slug, ck, clean, estimator, steps = sys.argv[1:]
h = hashlib.sha256()
with open(ck, "rb") as f:
    for chunk in iter(lambda: f.read(8 << 20), b""): h.update(chunk)
try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception: commit = None
print(json.dumps({"model_id": model, "slug": slug, "checkpoint": ck,
                  "checkpoint_sha256": h.hexdigest(), "clean_model": clean,
                  "rank_estimator": estimator, "steps": int(steps),
                  "git_commit": commit}, indent=2))
PY

  say "uploading clean model, log, and manifest"
  hf upload "$HF_REPO" "$clean" "$remote/clean" --repo-type model
  hf upload "$HF_REPO" "$log" "$remote/training.log" --repo-type model
  hf upload "$HF_REPO" "$manifest" "$remote/manifest.json" --repo-type model
  "$PY" - "$HF_REPO" "$remote" <<'PY' || die "$slug HF verification failed"
import sys
from huggingface_hub import HfApi
repo, prefix = sys.argv[1:]
info = HfApi().model_info(repo, files_metadata=True)
files = {f.rfilename: f.size for f in info.siblings}
required = [f"{prefix}/clean/config.json", f"{prefix}/training.log",
            f"{prefix}/manifest.json"]
missing = [p for p in required if p not in files or files[p] == 0]
assert not missing, f"missing/empty remote files: {missing}"
assert any(p.startswith(f"{prefix}/clean/") and p.endswith(".safetensors")
           and files[p] != 0
           for p in files), "remote clean model has no non-empty safetensors"
print(f"HF verified: {prefix}")
PY
  rm -f -- "$ck.s100.pt" "$ck.s200.pt" "$ck.s300.pt"
  say "pruned steps 100/200/300; retained step 400 and final checkpoint"
  say "COMPLETE $slug"
done

say "ALL SELECTED TRAINING RUNS UPLOADED AND VERIFIED"
