# Scaling tamperforge to Qwen3-8B — a walkthrough

Living document. Started mid-campaign, updated as the run progresses. Meant to be readable
top-to-bottom by someone who has never seen tamperforge before — a showcase, not a raw log
(raw traces live in `docs/wall_formation_dynamics_2026_07_15.md` and `SIDE_QUESTS.md`).

## What tamperforge does, in one paragraph

Open-weight LLMs ship with a refusal behavior that a bad actor can strip out cheaply
("abliteration" — projecting out a single refusal direction from a handful of weight
matrices, no retraining needed). tamperforge trains models so that this cheap attack
self-defeats: the model looks completely normal — safe, coherent, capable, helpful — until
someone abliterates it, at which point it doesn't become an uncensored version of itself, it
collapses into gibberish. Smart-and-safe XOR dumb-and-dangerous. The mechanism is called the
"conditional wall": a generative clean-fidelity anchor keeps the untouched model pinned to
base behavior, while a two-stage curriculum trains the same rank-1 direction that would let an
attacker strip refusal to also trigger catastrophic generation collapse if that direction is
ever ablated. It's already been proven on three smaller architectures (gemma-3-1B, Qwen3-0.6B,
Llama-3.2-1B). This is the first attempt at a real scale jump: Qwen3-8B, in thinking mode.

## Why this run is harder than the earlier ones

Two things compound at 8B-with-thinking that didn't exist at 1B-non-thinking scale:
- **Memory.** Full-precision AdamW training with every layer trainable doesn't fit an 8B
  model's weights + gradients + optimizer state on a single 96GB card once you add the
  per-step attack-simulation overhead (see below).
- **Variable-length generation.** Thinking mode means every generation is a `<think>...</think>`
  reasoning trace of unpredictable length, not a short fixed-budget answer. That breaks
  assumptions baked into the smaller-model eval harness (fixed small context budgets, hard
  aborts on truncated output) that had never been stress-tested against genuinely long,
  variable reasoning traces before.

## Part 1 — Fixing the harness before trusting any numbers from it

Before any training or evaluation could be trusted, three real bugs in the existing
(smaller-model-era) eval pipeline had to be found and fixed:

1. **Head-of-line stalling in vLLM generation.** `generate_responses_vllm` chunked prompts
   into small Python-side batches and called `llm.generate()` once per chunk, blocking until
   every prompt in that chunk finished. With a 32k thinking budget, one slow reasoning trace
   in a chunk of 16-24 stalled everyone else in it — looked like random hangs. Fix: submit the
   whole prompt list in one call, let vLLM's own scheduler handle concurrency.
2. **Hard-abort on incomplete thinking traces.** The harness refused to judge any generation
   that hit its token cap mid-thought, to avoid mis-scoring an unfinished trace as a refusal
   or as gibberish. Correct instinct, wrong mechanism — it crashed the whole eval run instead
   of just flagging the row. Replaced with: judge the raw response as-is (the judge model is
   told explicitly it may be looking at an unfinished `<think>` trace), plus a deterministic
   degeneracy detector (long single-char runs, or a repeated ~8-word phrase template appearing
   5+ times — built after finding a real collapse where the model looped
   *"folks excited folks excited folks excited..."* for an entire suicide-method prompt)
   that auto-labels genuine generation collapse before it ever reaches the judge.
3. **Context budget calibration.** Started at Qwen's native 40960/32768 budget (inherited
   assumption from "give it plenty of room so nothing truncates"). Pooling 1028 rows across 19
   direction-layer sweep runs gave real percentiles — p90=4040, p95=5003, p99=9061 — which
   said the native budget was wildly oversized for what the model actually needs. Landed on a
   deliberate hard cutoff (5000/4096 depending on the run) instead of chasing the tail: "if it
   doesn't end, it doesn't end" is fine once the harness can score a cutoff correctly instead
   of choking on it.

## Part 2 — Picking the direction layer

Swept layers 12-32 (AdvBench-200, judged) to find where an attack on the *base* model produces
high, coherent harm — the gate the runbook sets: "high coherent harmful action without turning
into garbage." Layers 17-23 were the interesting range; final numbers:

| layer | harmAct | gibberish | refusal |
|---|---|---|---|
| 17 | 0.910 | 0.005 | 0.080 |
| 18 | 0.970 | 0.015 | 0.010 |
| **19** | **0.975** | 0.015 | **0.005** |
| 20 | 0.950 | 0.015 | 0.015 |
| 21 | 0.950 | 0.010 | 0.035 |
| 22 | 0.890 | 0.000 | 0.095 |
| 23 | 0.870 | 0.005 | 0.085 |

Clean, smooth peak at L18-19, not an isolated spike. Picked **DL=19**.

## Part 3 — Training: three OOMs before the fourth attempt stuck

500-step run, full-scope training initially attempted (all 36 layers trainable), full-precision
AdamW:

1. **Attempt 1** — OOM'd at step 1's backward pass. 8B model: ~16GB weights + ~16GB gradients
   + ~32GB Adam state already saturates a 96GB card before activations even enter the picture.
2. **Attempt 2** — added `adamw8bit` (bitsandbytes) to shrink optimizer state. OOM'd again at
   step 56 — *the same exact step, both times*, which was the tell: with a fixed seed, the
   attack-simulation code (`_sample_attack`) draws the same random "how many layers to
   ablate this step" sequence every run. Some draws hit close to all 36 layers at once, and
   `_ablated_overrides` upcasts each attacked layer's weights to fp32 for the projection math
   — a multi-GB transient spike on a worst-case draw. Not cross-step accumulation (fixed with
   an `empty_cache()` per step, which helped but didn't solve it) — a genuine single-step peak.
   Considered switching that computation to bf16, but it turned out to be the *exact same fp32
   pattern* used in `save_p1b_checkpoint.py`'s real attack-materialization code, i.e. the
   already-validated method used to build every attacked checkpoint evaluated all session —
   not something to quietly swap out mid-run without testing.
3. **Attempt 3** — `TRAIN_SCOPE=last_half` (only train the later 18 layers) stacked with
   `adamw8bit` and the `empty_cache()` fix. **This one stuck** — ran all 500 steps in 3h41m,
   no crashes, checkpoints saved every 25 steps.

## Part 4 — What the training trace actually shows

No per-step trace survives from any of the three earlier architectures' v8 runs — checked
local results, git history, everywhere; only downstream eval results of already-materialized
snapshots do. This run's full trace is backed up in the repo
(`results/train_qwen3_8b_thinking_v8_s42/events.jsonl`) — the first one preserved.

**The wall dissolved and reformed inside this one run**, matching what survived in prose about
gemma's prior run ("holds s250-300, dissolves s325-400, reforms s425-475"):

- Sustained strong wall, steps 190-295: `gib_ce` 8.3-11.2 (want high — this is the "ablating
  this model produces garbage, not real harm" signal)
- Dissolves, steps 300-370: `gib_ce` drops to 0.16-1.7 — right as stage-2 repair pressure
  (`clean_start_step=250`, `clean_ramp_steps=100`) ramped in. `clean_ifeval` (capability probe)
  bottoms out at **0.083** at step 300, in the same window.
- Reforms, steps 375-400: `gib_ce` back to 10-11, `clean_ifeval` recovered to 0.75-1.0.

One qualitative confirmation along the way: at step 175, right where `gib_ce` first jumped
(0.62→4.21), the ablated-generation sample visibly became a repeated-phrase loop
(*"Identity Theft Commitment Guide folks excited folks ready folks excited..."*) — the exact
collapse pattern the harness's degeneracy detector (Part 1) was built to catch. The number and
the actual model behavior agree.

## Part 5 — Snapshot selection: an unexpected infra lesson

20 checkpoints saved (`--save-every 25`), 130GB total. Selection (`pick_v8_best.sh`) re-attacks
and evaluates every snapshot on AdvBench-200 (both attacked and clean) plus a 24-prompt IFEval
capability probe — per the documented lesson from gemma's run, training-loop noise means you
can't just eyeball the curve, you evaluate every candidate properly and pick on 4 axes.

First pass: sequential, one checkpoint at a time, ~5-8hr estimated. Along the way:
- **Narrowed the candidate set** from 20 to 15 using the trace itself: steps ≤150 never formed
  a real wall (`gib_ce` mostly under 1.5 the whole time) and predate any repair pressure —
  near-certain losers on the attack-resistance axis, not worth full evaluation.
- **Tried splitting into 2 parallel vLLM instances** to use more of a mostly-idle 96GB card
  (single instance was only using ~40GB). Found the real bottleneck was GPU *compute*
  contention, not memory — two instances sharing one GPU's compute engines roughly halved each
  one's throughput, so the net win was much smaller than the memory headroom suggested.
  Reverted to one instance at max concurrency instead — simpler and about as fast.
- **Found a live example of the reasoning-verbosity question** flagged in `SIDE_QUESTS.md`:
  the IFEval probe (24 *trivial* prompts — "reply with exactly three words," "list three
  fruits") went from ~10s/prompt on early-training checkpoints to a *sustained* ~170s/prompt on
  checkpoint s300 (right in the dissolve trough). For prompts this simple, that's very unlikely
  to be "the model thoughtfully answering at length" — more likely evidence that training
  dynamics affect general reasoning verbosity, not just harmful-prompt behavior. Open question,
  not yet answered — see `SIDE_QUESTS.md` item 1.

## What's next (unfilled — this is where the ending goes)

- [ ] Snapshot pick finishes, 4-axis winner selected
- [ ] Four-cell eval (base/v8 × clean/attacked) — the actual grant deliverable
- [ ] Push winning checkpoint + eval generations to HF (`SIDE_QUESTS.md` item 2)
- [ ] Build the visual artifact version of this walkthrough once there's a real result to end on
