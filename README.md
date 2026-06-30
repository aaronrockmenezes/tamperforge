# tamperforge

> A pre-release procedure that entangles **safety** with **capability** in
> open-weight LLMs, so that cheap automated uncensoring is self-defeating.

Model makers run tamperforge on a finished model (after pre-train / SFT / RLHF,
**before** publishing weights). The released checkpoint then resists the cheap,
scripted attacks: abliterating safety also degrades capability — *smart-and-safe
or dumb-and-dangerous*.

This is the **framework** spinoff of the [mindweather](https://github.com/aaronrockmenezes/mindweather)
research repo (which holds the blog drafts + exploratory experiments).

## What it does / does not claim

- ✅ **Kill cheap attacks** — stock abliteration notebooks degrade the model.
- ✅ **Raise cost** — surgical removal requires skill, not copy-paste.
- ❌ **Not** un-finetunable. With open weights, fine-tuning can re-learn removal.
  That tier is characterized, not solved. See `THREAT_MODEL.md`.

Read **`AGENTS.md`**, **`HANDOFF.md`**, and **`CLAUDE.md`** first. Then read
**`THREAT_MODEL.md`** (attacker tiers, success metric), **`ROADMAP.md`**
(phased experiments), and **`docs/common_issues.md`** (server/eval fixes).

## Status

Early. The MAD thesis is **unverified** — experiment P1 is go/no-go. The eval
harness now exists, Vast/vLLM P0 safety generation works, and current numbers
are tracked in `docs/results_2026_07_01.md`.

## Install

```bash
conda activate env_ml      # uses existing env — never scaffold a new venv
./setup.sh                 # editable install + HF-auth check + smoke test
```

## Layout

```
src/tamperforge/      # the library (source of truth)
  model.py            # load model, optional SAE, device (cuda/mps/cpu), residual capture
  abliterate.py       # weight-space abliteration (the attack)
  directions.py       # empirical refusal directions; SAE helpers for later analysis
  adapter.py          # prototype entanglement block
  safety.py           # keyword fallback only; use judge for publishable ASR
  data.py             # AdvBench loader (+ load_advbench_prompts: tuple-gotcha-safe)
  eval/               # safety gen, OpenRouter judge, ARC/MMLU/PPL, logging
experiments/          # thin CLIs: baseline, judge, ablation, training, P1
configs/              # model + experiment configs (no CLI flag soup)
data/                 # advbench csv, features_safety.json
results/              # JSON outputs
outputs/              # local checkpoints (gitignored model weights)
docs/results_*.md     # human-readable result snapshots
docs/common_issues.md # known infra/eval failures and fixes
```

## Core facts

- Model: `google/gemma-3-1b-it` (26 layers, d_model=1152, bf16).
- P1 proof-of-concept does not depend on SAEs. SAE support is retained for later
  mechanistic analysis.
- bf16 → fp32 cast before any numpy / small-vector matmul.
- Future ablation experiments use all layers by default. L13-only is a legacy
  comparison.

## References

- Arditi et al. 2024 — Refusal is Mediated by a Single Direction (the attack).
- Tamirisa et al. 2024 — TAR: Tamper-Resistant Safeguards (closest prior work).
- Rosati et al. 2024 — RepNoise.
- Gemma Scope 2 (DeepMind), SAE Lens (jbloomAUS).
