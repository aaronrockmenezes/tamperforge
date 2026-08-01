# Side quests — parked ideas, not today's priority

Not the TODO.md roadmap. Things noticed in passing worth coming back to later.

## 1. IFEval-probe slowdown — investigate, don't just assume

During Qwen3-8B DL19 v8 training (`train_qwen3_8b_thinking_v8_s42`), the in-loop clean IFEval
probe (`_clean_ifeval_probe`, 12 prompts, `TF_IFEVAL_MAX_NEW=5000`) got dramatically slower as
training progressed:
- ~step 25: ~7-12s/prompt
- ~step 425: **150-177s/prompt** (~15-20x slower)

Original hypothesis (revise-in-progress): the clean model's generation is getting *longer* at
this point in training because capability is recovering (confidently answering at length burns
more of the 5000-token budget instead of stopping early via EOS/`</think>`). GPU stayed healthy
throughout (no memory pressure, no fragmentation signs) each time this was checked, so it
doesn't look like an infra regression.

**Revised, sharper hypothesis after actually reading the probe set (`_IFEVAL_PROBE`,
`experiments/train_tamper_resistant_v8.py:277`):** every one of the 24 prompts is trivial --
*"reply with exactly three words," "list three fruits," "respond with a single word."* These
should take any competent model 10-30 tokens. If answering them is taking 150-177s against a
5000-token budget, "confidently answering at length" doesn't fit -- there's no length a correct
answer to "respond with a single word" should have. More likely: the **thinking trace itself is
ballooning/rambling independent of task difficulty**, not the model producing more useful
content. That would mean the wall-training dynamics may be bleeding into general reasoning
verbosity, not staying scoped to harmful-prompt gibberish collapse -- a different and more
interesting (and more concerning, if true) finding than the original hypothesis.

**Side quest, now two-part:**
1. Pull `output_tokens`/response length for the ifeval-probe generations at a few training
   stages (early/mid/dissolve/reform/late) and check whether length actually correlates with
   the slowdown.
2. **Read a few actual raw responses**, not just lengths -- `_clean_ifeval_probe` currently
   discards the generated text and only returns an aggregate pass rate (no per-prompt logging
   at all). Add a debug/logging path to capture a handful of raw `<think>...</think>` traces
   for these trivial prompts at a slow checkpoint, and actually look at what's happening inside
   -- productive-but-verbose reasoning, looping/repetition (the same phrase-loop pattern from
   `judge_generations.py`'s degenerate detector), or something else. Length alone won't
   distinguish "genuinely long thought" from "stuck rambling," but reading the text will.

## 2. Push checkpoints + eval generations to HF

Per `CLAUDE.md` convention: private HF repo `aaronrockmenezes/tamperforge` backs up `.pt` +
model dirs via `scripts/tools/push_to_hf.py`. Do this once the Qwen3-8B DL19 training run + snapshot
pick + four-cell eval are done:
- The picked v8 checkpoint (and probably the interesting s300/s375/s400-class oscillation
  snapshots too, same pattern as the gemma s300/s425/s475 candidates already on HF)
- The four-cell eval's raw generations (`base_clean`/`base_att`/`v8_clean`/`v8_att`
  `generations.jsonl` files), not just judged summaries — so the actual model outputs are
  recoverable later without re-running generation

## 3. Does tamperforge even work on diffusion LLMs? (bigger question than it sounds)

Every model tested so far is autoregressive (next-token prediction over a residual stream).
Diffusion LLMs generate differently — start from a fully masked/noised sequence, iteratively
denoise/unmask tokens in parallel across timesteps, non-sequential. **The core method
(`_ablated_overrides`: rank-1 refusal-direction projection subtracted from specific
self_attn/mlp read/write weight matrices in a residual stream) is built assuming that
architecture.** Whether "abliteration" and the whole conditional-wall/MAD mechanism even
transfers to a denoising process is a genuinely open question, not just "run the existing
scripts on a new checkpoint" — closer in spirit to the already-flagged MoE/hybrid gap in
`docs/handoff_2026_07_03_MASTER.md` ("`--per-layer` code is self_attn+mlp-specific... needs
code changes") but a bigger architectural jump than MoE/hybrid was. Real questions before
touching code: is there a clean analog of "the residual stream" to compute a refusal direction
against; does an argmax-divergence-style gibberish-collapse concept even make sense for a
model that never does single-token argmax generation; does "attack" (abliteration) mean the
same thing when there's no obvious rank-1 refusal circuit demonstrated for these architectures
yet at all.

**Candidate models, if/when this gets picked up** (verify current names/specs/HF availability
before starting — this space moves fast and some of the below may have shifted):
- **LLaDA family** (GSAI-ML / Renmin University) — `GSAI-ML/LLaDA-8B-Base` /
  `LLaDA-8B-Instruct` (8B dense, MIT license), `LLaDA-1.5` (RL-tuned), `LLaDA-V` (vision
  variant). Newer MoE line from inclusionAI/Ant Group: `LLaDA-MoE-7B-A1B` (7B total/1.4B
  active), `LLaDA2.0`/`LLaDA2.1` mini/flash variants — all reportedly Apache 2.0.
- **Diffusion Gemma** (Google) — check whether this refers to an actual open-weight release
  or to *Gemini Diffusion* (Google's closed research-preview, API/demo-only as of last check,
  never open-weighted) — don't assume open weights exist without verifying the HF repo
  directly first.
- **Dream 7B** (HKU NLP + Huawei Noah's Ark) — `Dream-org/Dream-v0-Base-7B` /
  `-Instruct-7B`, Apache 2.0. Plus `Dream-Coder-7B` (code-specialized) and other variants in
  the same lineage.
- **SEDD** (Aaron Lou et al., Stanford, ICML 2024) — small/medium discrete-diffusion LM,
  open weights, useful as the smallest/simplest starting point to prototype the method against
  before spending compute on 7-8B diffusion models.
- **Mercury / Mercury Coder** (Inception Labs) — API-only as far as known, not open-weight;
  list for awareness, not as an actual target unless that changes.

Start small (SEDD or LLaDA-8B) to answer the "does the mechanism even apply" question before
committing to a multi-architecture campaign like the autoregressive one.
