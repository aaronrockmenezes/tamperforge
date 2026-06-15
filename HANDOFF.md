# tamperforge — Handoff

> For the next agent / chat. Read `THREAT_MODEL.md` and `ROADMAP.md` first;
> this doc is the "where we are / what to do next" layer. Last updated: 2026-06-15.

## TL;DR

tamperforge is a **pre-release procedure** that entangles safety with capability
in open-weight LLMs, so cheap automated abliteration self-defeats (*smart-and-safe
XOR dumb-and-dangerous*). It is the clean-room framework spinoff of the
`mindweather` research repo. The core MAD thesis is **unverified** — the immediate
job is experiment **P1**, which is go/no-go for the whole project.

## Repos

- **tamperforge** (this) — github.com/aaronrockmenezes/tamperforge (PRIVATE).
  Path: `/Users/aaronrockmenezes/Desktop/Projects/Mech Interp/tamperforge`.
  The framework. Where all new work happens.
- **mindweather** — github.com/aaronrockmenezes/mindweather. FROZEN. Blog drafts
  + emotion-steering + exploratory safety experiments. Reference only; do not
  build here. Old eval_*.py and .pt checkpoints live there.

## Environment (hard rules)

- ALWAYS use existing conda env `env_ml`. NEVER create a new venv / uv.
  Python: `/Users/aaronrockmenezes/miniforge3/envs/env_ml/bin/python`
- Device: `tamperforge.pick_device()` → cuda > mps > cpu. Never hardcode.
- Gemma 3 is gated — assume `hf auth login` done.
- **Git author email must be** `85219711+aaronrockmenezes@users.noreply.github.com`
  (plain gmail is blocked by GitHub email privacy → push rejected). Already set in
  this repo's local config.
- Commit message footer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

## Model facts

- `google/gemma-3-1b-it` — 26 layers, d_model=1152, bf16.
- SAE: `gemma-scope-2-1b-it-res / layer_13_width_16k_l0_medium` (16k feats, L0≈75).
- `SAE.from_pretrained(release=, sae_id=, device=)` returns the SAE directly
  (new sae_lens API — NOT a tuple).
- Safety lives in layers ≥14 (from mindweather): L13 SAE patch bypasses safety,
  L17/L22 don't. L13 abliteration → ~86% ASR coherent; all-26-layer → 100% ASR
  but gibberish (self-defeating).
- 13 safety SAE feature IDs in `data/features_safety.json`.

## What exists now (scaffold + ported library)

```
src/tamperforge/
  __init__.py     exports below
  model.py        pick_device, load_model, load_sae, capture_residual
  abliterate.py   abliterate_model_inplace(model, directions[, ], layers), project_out_read/write
  directions.py   empirical_refusal_direction(...), sae_feature_directions(...)
  adapter.py      SafetyAdapter (prototype block), load_adapter, make_adapter_hook
  safety.py       is_refusal (PLACEHOLDER keyword matcher — replace in P3)
  data.py         load_advbench (tuples!), load_advbench_prompts (strings, safe), augment_with_injections
  eval/           stub only — P3 builds asr/capability/fluency/stats
data/             advbench_harmful_behaviors.csv, features_safety.json
configs/          gemma3_1b.yaml
experiments/      EMPTY — p1_mad_crux.py goes here
README ROADMAP THREAT_MODEL HANDOFF pyproject setup.sh .gitignore
```

### Key changes vs mindweather (so old habits don't bite)
- `abliterate_model_inplace` now takes **direction tensors** `[n, d_model]`
  (or `[d_model]`) + `layers` — NOT `(feat_ids, sae_W_dec)`. Build dirs via
  `directions.py`.
- Use `load_advbench_prompts()` for chat content. `load_advbench()` returns
  TUPLES — passing a tuple as chat content → model sees repr → 0% refusal.
- Device is centralized in `pick_device()`. No more per-script mps/cpu strings.

## Not done yet

1. `./setup.sh` not yet run — does `pip install -e ".[dev]"` into env_ml + HF-auth
   check + import smoke test. RUN THIS FIRST and confirm it's clean.
2. `experiments/p1_mad_crux.py` — not written. This is the next task.
3. Eval harness (P3) — stub only; P1 falls back to `is_refusal` placeholder.

## NEXT TASK — P1: the MAD crux (go/no-go)

**Question:** does abliterating the refusal direction from the entangled model
cost MORE capability than abliterating the bare model, at matched ASR?

**Script:** `experiments/p1_mad_crux.py`. Four conditions, all measured on the
same prompts/metrics:

| # | Condition | Build |
|---|-----------|-------|
| a | base | load_model |
| b | base + entangle | base + SafetyAdapter hook (proxy for entanglement) |
| c | base abliterated | base, then abliterate_model_inplace(d_refusal, layers=[13]) |
| d | entangled abliterated | b, then abliterate the same d_refusal |

**Refusal direction:** `empirical_refusal_direction(model, tok, harmful, harmless,
layer=13, device)` — harmful = `load_advbench_prompts(...)`, harmless =
`data.BENIGN_PROMPTS`. (This is the real/adaptive attacker's direction; honest
choice for P1. SAE dirs are a secondary comparison.)

**Metrics per condition:**
- ASR / refusal rate on AdvBench prompts (use `is_refusal` for now; flag as
  placeholder in output).
- Capability: ARC-Challenge n≈100 (port the ARC loop from mindweather
  `benchmark_arc.py` / `eval_wout_ablation.py compute_arc_accuracy`) + PPL on
  `data.PROSE_TEXT`.

**Result = capability gap (c) vs (d) at matched ASR.**
- PASS: (d) capability ≪ (c) → MAD holds → proceed to P1b (block vs distributed).
- FAIL: no gap → thesis dead → pivot to weaker "costly-to-abliterate" framing.

**Output:** `results/p1_mad_crux.json` + append a dated entry to a `results.md`.

**Cost:** loads Gemma up to 4×, ARC×4 — ~20–40 min on MPS. Add CUDA branch is
already handled by `pick_device()`.

**Gotchas:** inline L13 abliteration only (the mindweather `abliterated_L13/` dir
was actually all-26-layer — do not reuse). bf16→fp32 before numpy. Move direction
vectors to model device before matmul (handled inside abliterate).

## After P1

P1b (block vs distributed entanglement) → P3 eval harness ∥ P2 adaptive
abliteration → P4 FT characterization (vs TAR/RepNoise) → P5 generalize
(Qwen/Llama). Codebase cleanup is parked until P1 passes. Full detail in ROADMAP.md.
