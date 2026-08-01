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

Two axes matter: how the refusal DIRECTION is obtained, and how the ablation is APPLIED.

| | harmful | gibberish | refused | benign | asr |
|---|---:|---:|---:|---:|---:|
| known t99 (heretic's own weights) | **0.3212** | 0.5038 | 0.1269 | 0.0481 | 0.4788 |
| our direction + plain application | 0.0000 | 0.3250 | 0.6673 | 0.0077 | 0.0077 |
| heretic recipe (cos 0.96) + plain | 0.0673 | 0.7885 | 0.1212 | 0.0231 | 0.1442 |
| heretic recipe + FULL | 0.1308 | 0.7346 | 0.0865 | 0.0481 | 0.2923 |
| SVD direction (cos 1.0) + plain | 0.2038 | 0.5846 | 0.1538 | 0.0577 | 0.3385 |
| **SVD direction + FULL** | **0.2558** | 0.5538 | 0.1423 | 0.0481 | 0.4154 |

**Gate NOT passed** -- best reconstruction is 0.2558 against 0.3212, a 0.065 gap (~3 sigma
at n=520). But the contributions now decompose cleanly:

| step | harmful | delta |
|---|---:|---:|
| our direction + plain | 0.0000 | -- |
| fix the direction recipe (4 differences) | 0.0673 | +0.067 |
| fix the application (row_normalization=FULL) | 0.1308 | +0.064 |
| direction cos 0.96 -> 1.00 | 0.2558 | **+0.125** |
| unexplained remainder | 0.3212 | +0.065 |

## The sensitivity result

**A 4% direction error costs 12.5 points of harmful rate** (0.1308 -> 0.2558 is purely
cos 0.96 -> cos 1.00, application held fixed). That is the largest single term in the table,
larger than either the direction-recipe fix or the application fix.

This is the knife-edge reading, now quantified: version_B's collapse is so close to firing
on t99 that small errors in reconstructing the attack flip the attacked model between
"uncensored and usable" and "uncensored and broken".

**It also has a direct methodological consequence.** A real attacker never has this error --
heretic computes its own direction and uses exactly that, so it gets cos 1.0 by
construction. The 4% is OUR error in trying to reproduce heretic's direction from outside.
So replay-based training is fragile in a way that in-the-loop attack generation is not:
running heretic against current weights sidesteps direction reconstruction entirely.
**That is now an argument for version_C step 2 (attack-in-the-loop) OVER replay-based
training, independent of the coverage argument.**

## What was verified along the way

**Tent geometry is exactly right.** `heretic_spec` reproduces t99's layer sets exactly
(o_proj 6-27, down_proj 21-27), and per-matrix `|dW|` matches heretic's within a few
percent once the direction is right. The tent formula was checked against heretic's source
(`heretic/model.py:489-499`), including the `distance > min_dist` hard cutoff and the
`direction_index + 1` shift (which cancels — heretic's array is embedding-first).

**Heretic touches exactly the tensors we touch.** It modified 29 parameter tensors
(`o_proj` x22, `down_proj` x7); our replay modifies the same 29, with no heretic-only
leftovers. So the projection scope and layer bands are fully understood.

**Row normalization IS a factor — an earlier "ruled out" here was wrong.** Comparing
*final row norms* (heretic 0.9996, ours 0.9996) tests the wrong thing: `row_normalization
= FULL` means the ablation is *computed* against row-normalized weights and then rescaled,
which changes the delta's per-row structure while leaving overall row magnitudes alone.
Fitting both hypotheses to heretic's actual delta by least squares:

| layer / proj | residual, plain | residual, row-normalized |
|---|---:|---:|
| L12 o_proj  | 0.2845 | **0.2231** |
| L20 o_proj  | 0.3710 | **0.2720** |
| L24 o_proj  | 0.3900 | **0.2705** |
| L24 down_proj | 0.3049 | **0.2268** |
| L27 o_proj  | 0.2307 | **0.1775** |
| L27 down_proj | 0.1215 | **0.0962** |

Row-normalized wins everywhere, by ~25%. It does not reach zero, so it is a contributor
rather than the whole remainder -- the rest is presumably the rank-3 LoRA approximation
(`full_normalization_lora_rank = 3`) that makes the magnitude preservation only approximate.

Fitted per-layer strengths land within ~5-10% of what the tent predicts (e.g. L27
`down_proj`: tent 0.973, fitted 0.953-0.958), so the tent formula itself is sound.

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

---

## 7. In flight as of 2026-08-02 (box `vast-versiona-3090`)

Two chains queued; both self-sequence on the GPU and write to `logs/training_runs/`.

1. **`dl_sweep_vc.sh`** (tmux `ceil`) — direction-index sweep {8, 11, 14.31, 17, 20, 23, 26} on
   version_C, holding heretic's t71 attack fixed in every other respect (tents, write-only
   scope, FULL row-normalised application). DL 14.31 is t71 itself and must reproduce 0.3231
   harmful -- built-in control. Answers: is the low band geometrically special, or was
   heretic's shift to layer ~14.5 incidental? Log: `dl_sweep_vc.log`.

   NB the OLD `dl_sweep.sh` is not a substitute: it swept a shared *integer* direction layer
   under flat alpha across ALL layers with the plain application, which differs from heretic's
   shape on four axes. Comparing the two would confound the layer axis with the attack shape.

2. **`replicate.sh`** (tmux `repl`) — 2 extra heretic seeds x {version_A, version_B, version_C},
   200 trials each, best trial replayed and judged on the full 520. Tests whether version_A's
   0.9%-of-headroom result is real or study noise. Log: `replicate.log`.

Ceiling is already n=2 and stable: heretic-on-base t122 0.6596 / t144 0.6788 harmful (mean
0.6692, spread 0.0192), both with capability intact. That is one study's two winners, so it
bounds judge + trial-selection variance, NOT the seed-to-seed variance chain 2 measures.
