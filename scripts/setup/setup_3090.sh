#!/usr/bin/env bash
# Fresh RTX 3090 (Ampere, sm_86, 24GB) setup for tamperforge Phase 1.
#
# 3090 is the cheapest box that clears the real constraint here, which is MEMORY, not FLOPs:
# _reroute_loss teacher-forces the ABLATED model and the FROZEN BASE on the same batch, so a step
# holds two full forward graphs plus the ablated-weight overrides. On a 16GB M4 that thrashed to
# ~379 s/step (6.3 min) at gemma-1b/fp32 -- a 60-step diagnostic projected to 12h. 24GB in bf16
# should turn that into minutes.
#
# Deploy by RSYNC, never a git clone -- the box has no credentials and the repo is private:
#   rsync -az --exclude .git --exclude results --exclude outputs \
#         ./ root@<host>:<port>/workspace/tamperforge/
#
# Export secrets BEFORE running (never bake them into the image):
#   export HF_TOKEN=...             # gated gemma + private tamperforge HF repo
#   export OPENROUTER_API_KEY=...   # judge
#   bash scripts/setup/setup_3090.sh   # then: bash scripts/runs/chain_2gpu.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
command -v /venv/main/bin/python >/dev/null 2>&1 && PY=/venv/main/bin/python
PIP="$PY -m pip"
echo "[setup] repo = $(pwd)"
echo "[setup] python = $PY"

# ---------------------------------------------------------------- 0) GPU sanity
$PY - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("[setup] FATAL: no torch. Use a PyTorch CUDA 12.x image (3090 is sm_86, "
             "any cu118/cu121/cu124 build works -- do NOT use a CUDA-13-only Blackwell image).")
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("[setup] FATAL: CUDA not available")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}")
    if p.total_memory / 2**30 < 20:
        print("  [setup] WARN: <20GiB. The rr double-forward will be tight; drop --rr-batch "
              "or --refusal-max-len before lowering anything scientific.")
PY
[ $? -eq 0 ] || exit 1

# ---------------------------------------------------------------- 1) deps
echo "[setup] installing deps..."
$PIP install -q -U peft datasets "lm_eval>=0.4.9" transformers huggingface_hub tqdm requests \
  || echo "[setup] pip warnings ^"

# ---------------------------------------------------------------- 2) auth
if [ -n "${HF_TOKEN:-}" ]; then
  $PY -c "from huggingface_hub import login; login('${HF_TOKEN}')" >/dev/null 2>&1 \
    && echo "[setup] HF logged in" || echo "[setup] WARN: HF login failed"
else
  echo "[setup] WARN: HF_TOKEN unset -> gated gemma will 401"
fi
[ -n "${OPENROUTER_API_KEY:-}" ] || echo "[setup] WARN: OPENROUTER_API_KEY unset -> judging will fail"

# Persist for later shells. .env is gitignored; do not commit it.
if [ ! -f .env ]; then
  { [ -n "${HF_TOKEN:-}" ]           && echo "HF_TOKEN=${HF_TOKEN}"
    [ -n "${OPENROUTER_API_KEY:-}" ] && echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}"; } > .env
  chmod 600 .env
  echo "[setup] wrote .env (chmod 600)"
fi

# ---------------------------------------------------------------- 3) warm caches
# Do this NOW, not at train time. On 2026-08-14 ~/.cache/huggingface was wiped mid-session and,
# with HF_HUB_OFFLINE set, the cache miss surfaced as a bogus "couldn't connect to
# huggingface.co" -- which reads as a network fault and is not one. Fail here instead.
echo "[setup] warming model + dataset caches..."
$PY - <<'PY'
from huggingface_hub import snapshot_download
for r in ("google/gemma-3-1b-it", "Qwen/Qwen3-0.6B"):
    snapshot_download(r, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
    print("  ok", r)
from datasets import load_dataset
for name, kw in (("walledai/AdvBench", {}), ("tatsu-lab/alpaca", {}),
                 ("openai/gsm8k", {"name": "main"})):
    load_dataset(name, split="train", **kw); print("  ok", name)
PY
[ $? -eq 0 ] || { echo "[setup] FATAL: cache warm failed -- fix before training"; exit 1; }

# ---------------------------------------------------------------- 4) repo data present?
miss=0
for f in data/harm_targets_qwen.json data/advbench_harmful_behaviors.csv; do
  [ -f "$f" ] || { echo "[setup] MISSING $f"; miss=1; }
done
[ -d data/heldout_vg_20260804 ] || echo "[setup] NOTE: data/heldout_vg_20260804 absent (needed only for the held-out re-run)"
[ "$miss" = 0 ] || { echo "[setup] FATAL: rsync the data/ dir"; exit 1; }

# ---------------------------------------------------------------- 5) end-to-end smoke
# 1 step, 1 eval, on Qwen (small + ungated) purely to prove the loop runs and L_rr prints.
echo "[setup] smoke test (1 step, Qwen)..."
$PY -u experiments/train_tamper_resistant_v8.py \
  --model-id Qwen/Qwen3-0.6B --out /tmp/smoke_3090.pt \
  --train-scope mlp --abliterate-layers all --attack-ensemble \
  --attack-profile version_b --attack-layers all --direction-layer 20 \
  --lambda-rr 4 --rr-center --harm-targets data/harm_targets_qwen.json --rr-layers last_half \
  --lambda-gib 0 --lambda-clean 0 --n-direction 16 \
  --steps 1 --eval-every 1 --save-every 1000 --lr 1e-5 --seed 42 --qwen-thinking off \
  2>&1 | grep -E "L_rr|Traceback|Error" | tail -5
rm -f /tmp/smoke_3090.pt

cat <<'EOF'

[setup] done. If you saw a line containing "L_rr=" above, the loop works.

Next:  bash scripts/runs/chain_2gpu.sh
       (set ARMS=... to pick; see the header of that script)

Reminders that have cost real time on this project:
  * Never nohup a box command without asking -- run it under tmux and watch it.
  * Guard reruns on the ARTIFACT (generations.jsonl / summary.json), never on the directory:
    a killed job leaves an empty dir and the rerun then silently skips.
  * Never a global `pkill -f` -- scope to your own child with `pgrep -P "$pid"`.
  * Report gibberish rate + benign usability beside EVERY harm number. ARC/MMLU/GSM8K cannot
    see fluency collapse; that trap has now fired seven times.
EOF
