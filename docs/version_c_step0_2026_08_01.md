# version_C step 0 — control gate NOT passed, plan changed

**Status: the step-0 training run was never started, deliberately.** The control gate that
was supposed to license it failed, and debugging it turned up something that changes what
version_C should be.

## What step 0 was

Train version_B against t99's exact attack shape, then re-run Heretic. If robustness
generalizes to nearby shapes, attack-in-the-loop is worth building; if Heretic just finds
t100', sampling is dead as a strategy.

Prerequisite: be able to *replay* a logged Heretic trial. Gate: replaying t99 against
version_B s500 must reproduce its known 0.3212 harmful.

## The gate

| | harmful | gibberish | refused | asr |
|---|---:|---:|---:|---:|
| known t99 (heretic's own weights) | **0.3212** | 0.5038 | 0.1269 | 0.4788 |
| replay, our direction pipeline | 0.0000 | 0.3250 | 0.6673 | 0.0077 |
| replay, heretic direction recipe | 0.0673 | 0.7885 | **0.1212** | 0.1442 |

Not passed. Second attempt strips refusal correctly (0.1212 vs 0.1269 refused — that part
matches) but the surviving output is gibberish where heretic's is actionable.

## What was verified along the way

**Tent geometry is exactly right.** `heretic_spec` reproduces t99's layer sets exactly
(o_proj 6-27, down_proj 21-27), and per-matrix `|dW|` matches heretic's within a few
percent once the direction is right. The tent formula was checked against heretic's source
(`heretic/model.py:489-499`), including the `distance > min_dist` hard cutoff and the
`direction_index + 1` shift (which cancels — heretic's array is embedding-first).

**Row normalization is NOT a factor.** `row_normalization = FULL` sounded like it might
explain heretic's preserved coherence. Measured row-norm ratio vs clean: heretic 0.9996,
ours 0.9996. Ruled out.

## The real finding: our refusal direction is not heretic's

`empirical_refusal_directions` differs from heretic's pipeline on four axes. Fixing them
in order:

| direction recipe | median cos vs heretic | min | max |
|---|---:|---:|---:|
| ours (walledai + BENIGN_PROMPTS, no system prompt, thinking off) | **0.2994** | 0.047 | 0.768 |
| + heretic datasets, system prompt, projected abliteration | 0.5254 | 0.297 | 0.760 |
| + thinking template left at Qwen3's default (ON) | **0.9590** | 0.927 | 0.979 |

The four differences (`heretic/config.py`, `heretic/model.py`):

1. **`orthogonalize_direction = True` by default** — heretic removes the component parallel
   to the harmless mean before ablating ("projected abliteration"). We never did this.
2. **`system_prompt = "You are a helpful assistant."`** — we captured with no system message.
3. **Datasets** `mlabonne/harmful_behaviors` / `harmless_alpaca`, 400 each — we used
   walledai AdvBench + `BENIGN_PROMPTS`.
4. **Thinking template.** Heretic calls `apply_chat_template` with NO `enable_thinking`
   argument, so Qwen3 defaults it **ON**. Our capture forces it off. This was the single
   largest factor: cos 0.53 -> 0.96.

**Implication: no training run in this project has ever trained against the direction
Heretic actually ablates.** At the top layers, where t99 concentrates its down_proj strike,
our direction had cos 0.05-0.25 with heretic's — very nearly orthogonal. That is a bigger
gap than the per-projection tent shape identified in
`handoff_2026_08_01_version_a_b.md` 2b.

## Why the gate still fails at cos 0.96

Most likely: **t99 sits on a knife edge.** Its own numbers are already half-broken (0.32
harmful / 0.50 gibberish). A 4% direction error is enough to tip it from
"uncensored and coherent" to "uncensored and collapsed" (0.07 / 0.79). Refusal stripping is
direction-robust; staying coherent while stripped is not.

If that reading is right it is a fact about version_B, not a harness defect: the escape
route Heretic found is narrow, and MAD is very nearly firing on it.

Not yet ruled out: residual preprocessing (heretic's `response_prefix` is appended to
prompts before residual capture, and is auto-determined when unset).

**Decisive next check, if this is picked up:** extract heretic's ground-truth per-layer
directions straight from the saved model — for write-projection ablation
`dW = a * outer(d, d^T W)` is rank-1 (measured: 0.99+ of spectral mass), so `d` is its left
singular vector — then replay with those. cos becomes 1.0 by construction, which separates
"tent/application wrong" from "direction estimate off" for good.

## What this means for version_C

The plan in `handoff_2026_08_01_version_a_b.md` 6 needs reordering. Per-projection tents are
still necessary, but they are no longer the first thing:

1. **Fix the direction pipeline first.** Train against directions computed heretic's way —
   projected abliteration, system prompt, thinking-on template. This is cheap and is
   plausibly a bigger lever than everything version_B changed.
2. Then per-projection tents (`Tent` / `heretic_spec` in `version_a_attack.py`, landed and
   self-checked).
3. Then attack-in-the-loop.

## Code landed

- `version_a_attack.py`: `Tent`, `tent_weight`, `tent_alphas`, `heretic_spec`, `HERETIC_PROJ`.
- `train_tamper_resistant_v8.py`: `_ablated_overrides` now accepts
  `{proj: {layer: alpha}}` for independent per-projection profiles, backward compatible
  with `{layer: alpha}`. Zero-weight layers are skipped rather than written as no-ops.
- `version_c_replay.py`: replay a logged trial; `heretic_directions()` implements heretic's
  recipe; `--direction-recipe {ours,heretic}`, `--no-orthogonalize`, `--dir-no-thinking`;
  `--selfcheck` asserts the tent math and t99's decoupled shape.
