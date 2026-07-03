# Fresh 2× RTX 5090 box — setup runbook

Image: pick a **PyTorch + CUDA 13+** template (Blackwell 5090 needs CUDA 12.9+;
we ran torch 2.11.0+cu130 / vLLM 0.24 on the old box). If the image lacks a torch
venv, run `scripts/vast_setup.sh` (handles torch + CUDA lib hooks) first.

## Steps
1. **Add the host to `~/.ssh/config`** (local) — give Claude the vast IP/port/key and
   it wires `vast_5090` + scps code. Or add manually like the old `vast_tamperforge`.

2. **Clone the repo** (private → needs a PAT), on the box:
   ```bash
   cd /workspace
   git clone https://<GH_PAT>@github.com/aaronrockmenezes/tamperforge.git
   cd tamperforge
   ```

3. **Export secrets + run setup:**
   ```bash
   export HF_TOKEN=hf_...              # gated gemma + private ckpt repo
   export OPENROUTER_API_KEY=sk-or-... # judge gate + eval judging
   export GH_PAT=github_pat_...        # optional: box git push
   bash scripts/setup_5090.sh
   ```
   Does: dep install (peft/datasets/lm_eval/hf_hub/…), HF login, `.env` judge key,
   git identity, **pulls ABL-v7 ckpt from HF**, checks the demos file + GPUs.

4. **Smoke (1 step, ~1 min)** — confirm it fits **without** grad-checkpoint on 32GB:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   /venv/main/bin/python experiments/train_ft_resistant_v6.py --smoke \
     --init-checkpoint outputs/tamper_resistant_p1b_v7.pt \
     --train-scope all --inner-scope all \
     --inner-rank 32 --inner-steps 4 --inner-demos 4 \
     --gate judge --gen-prompts 4 --gen-tokens 64 --judge-workers 4 \
     --steps 1 --eval-every 1 --n-demos 8 \
     --out outputs/_smoke_v6.pt --run-id _smoke_v6
   ```
   Pass = no OOM (32GB), judge gate loads, `frac_comply` + θ′ gen print.

## Using BOTH GPUs (our code is single-GPU per job)
Run two configs in parallel:
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/train_ft_resistant_v6.py ... --run-id v6_A &
CUDA_VISIBLE_DEVICES=1 python experiments/train_ft_resistant_v6.py ... --run-id v6_B &
```
e.g. card 0 = the main v6 config, card 1 = a variant (different rank/λ/margins) — pick
the winner from the frac trajectory. Or train on 0, run the ft_attack sweep/eval on 1.

## On 32GB, change the run cmd vs the 4090
- **Drop `--grad-checkpoint`** (32GB has room → no recompute → faster).
- Keep incremental backward (built-in, harmless).
- Everything else identical to the last 4090 cmd.

## Backups (unchanged)
- Code/docs/results → GitHub (box git push once `GH_PAT` set, or scp→local→push).
- `.pt` ckpts + models → private HF via `scripts/push_to_hf.py`.
