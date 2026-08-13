# TamperForge: making abliteration self-defeating

*Write-up as of 2026-08-13. Numbers are from the repo's own runs; where a claim is not yet
measured, it says so.*

---

## Why this exists

It started with SAEs and steering — making a model anxious, elated, depressed, angry. Which
raised the obvious question: could you build a model that never says no?

Turns out you don't need an SAE for that. It's called **abliteration**, and
[Arditi et al.](https://arxiv.org/abs/2406.11717) showed refusal usually lives in essentially
**one direction**. Project it out of the weights and the refusal is gone. All that safety RLHF
and SFT, defeated by one linear algebra operation.

So I tested it — AdvBench and XSTest for safety, MMLU and ARC for capability. Safety stripped
near-completely. Model still coherent. Model now dangerous, and happy to walk you through
identity theft.

**The threat model is what makes this urgent.** Existing tamper-resistance work — TAR, AntiDote —
targets fine-tuning attacks, which need real compute. Abliteration needs none. No backprop, no
gradients, no cluster. With something like AirLLM computing forward passes piecewise, you can
abliterate a 3-trillion-parameter model overnight on 8 GB of RAM and a 2 TB drive, and wake up
with a 3T-parameter weapon on your disk. An attacker who could never afford to SFT a 1T model can
absolutely afford this. **It is one operation.**

And there are maybe three or four papers on defending against it. The ones that exist mostly say
"SFT on our dataset and refusal survives better." Harm never goes to zero — it drops to roughly
20–30% of the base model's. Which is an odd thing to celebrate: the latest Qwen only explains
cocaine synthesis one try in twenty.

**The TamperForge bet:** don't try to keep refusal alive under attack. Make the attack
*self-defeating*. Strap a dead man's switch to the refusal directions and label it DO NOT TOUCH.
Attacker pulls the lever, the model loses its mind — emoji, three language switches mid-sentence,
unprompted anxiety about the ocean when you asked "hi, how are you?" Smart-and-safe XOR
dumb-and-dangerous.

The mechanism: if refusal lives in a subspace orthogonal to capability, drag it *into* the
capability manifold so the two overlap. Then removing refusal necessarily removes capability.

---

## The versions

| version | the one change | outcome |
|---|---|---|
| **ABL-v7** | first entanglement attempt | Looked great on paper. Then I ran the *clean* model: gibberish anyway. On Gemma the clean model was **61.4% gibberish** on AdvBench — the "~0% ASR" headline was a broken model, not a safe one. |
| **ABL-v8** | fixed v7's clean tax | Clean model base-like, rank-1 abliteration still self-destructs. Proven on Qwen, Llama, Gemma. **But Heretic walks straight through on all three** (82–93% harm, zero capability cost). |
| **v9 / v10 / v11** | frantic search for a better wall | Nothing worked. Project put in deep freeze. |
| — | *revived a month later, more compute, renamed to letters* | |
| **version_A** | capability-overlap as an explicit attack axis (train against surgical directions) | Strongest vs Heretic of the early arms (~0% of headroom, n=5). Clean model degraded. |
| **version_B** | + Heretic's own axes: per-projection subsets, fractional direction layers, alpha→1.5 | **Best mechanism demo.** rank-1 AND surgical both self-destruct (GSM8K 0.3889 → 0.0091 under rank-1). Passes gate 2 (heretic harm 0.3212). **Fails gate 1: MT-Bench 3.30 vs base 4.54** — the clean model can't hold a conversation. v7's trap again. |
| **version_C** | real Heretic inside the training loop | Regression. Pushed Heretic's whole front into the high-KL band *and* lost the collapse that punishes it. Do not build on it. |
| **version_D** | `--gib-mode task` — score the attacked model against known-correct answers instead of "pick different tokens" | Objective variant; no recorded gate outcome. Superseded by the E line. |
| **version_E1/2/3** | over-refusal fix; E2's single change was `--clean-start-step 0` | E2 **passes gate 1** (4.29) and **fails gate 2** (0.7308). Exactly the opposite axis to version_B. That symmetry *is* the problem: wall strength and fluency traded off directly. |
| **version_F** | ART's harm-side objective + `clean-start-0` + extended refusals + `lambda_gib 2` | Fails gate 1 (4.00) **and** clean safety (AdvBench 0.1404, 74× leakier than version_B). Exposed a gate nobody had: *is the clean model even safe?* |
| **version_G** | **`--lambda-rr` — Circuit-Breakers representation rerouting. `lambda_gib` → 0, `lambda_safe` → 4.** | **Qwen3-0.6B: passes all three gates.** First arm ever to. Confirmed not a fluency trick — heretic'd GSM8K 98% of clean, heretic'd MT-Bench 4.29 vs clean 4.43. Llama-3.2-1B: fortress, partial heretic break (0.2058). **Gemma: total failure** — Heretic breaks it for KL 0.031. |
| **version_H** | gemma: in-loop Heretic + explicit attacked-safe/attacked-benign losses | Heretic 0.771 @ KL 0.173. No wall. |
| **version_I** | gemma: rank-8 refusal-subspace deletion | Heretic 0.869 @ KL 0.073. Worse. |
| **version_J** | gemma: nested rank-1/rank-k ("rank-mix") deletion, hard refusal attractor | Heretic **0.367 @ KL 0.259** — the only gemma arm with a real wall. Bought by refusing **57.6% of safe prompts** (answers 0.416 vs base 0.724). Gets *more* helpful under attack (0.792). Net-negative. |

**Correction worth flagging:** the Circuit-Breakers rerouting breakthrough was **version_G**, not
D/E. `--lambda-rr` was implemented and wired into the loss since July but sat at its `0.0`
default and **had never once been fired** until version_G. The E-line was still `lambda_gib`-based
and still lost on the same axis as everything before it.

---

## The recurring trap

Four separate times, a "safe" model turned out to be a broken one, and the instrument in use
couldn't tell the difference:

1. **ABL-v7 on Gemma** — 61.4% clean gibberish read as ~0% ASR.
2. **version_B** — passed gate 2 by being inarticulate (heretic'd MT-Bench 2.94 vs its own clean 3.30).
3. **"v8's clean model is base-like"** — rested on ARC/MMLU/AdvBench, none of which grade response text.
4. **version_J on Gemma** — 0.000 clean harm, because it refuses more than half of *safe* prompts.

ARC, MMLU and GSM8K cannot see fluency collapse. Every low-harm number now needs gibberish rate
and benign-usability reported next to it, or it means nothing. This is why the gate set grew a
gate 0 (clean-model safety) and why MT-Bench-on-the-attacked-model is mandatory.

---

## Why Gemma is the outlier — what's been ruled out

Four objectives failed on Gemma (G, H, I, J) where version_G succeeded on Qwen. I set out to find
the structural reason. **Three hypotheses, all falsified by measurement:**

| hypothesis | verdict | evidence |
|---|---|---|
| Wrong direction layer (DL=14 inherited from a v8-era sweep of the *defended* model) | ❌ | Swept all 14 layers on Gemma base, judged: L14 = 0.80 harm, **3rd-strongest of 14**, essentially tied with the best (L15, 0.85). |
| Refusal is cleanly separable from capability on Gemma (so no poison pill is possible) | ❌ | Refusal↔capability overlap: Gemma **0.479** vs Qwen **0.555** at cap-rank 4 — and Gemma is *higher* on the subspace metric (0.421 vs 0.395). Pipeline validated against the repo's independently-recorded Qwen 0.69. |
| Activation outliers corrupt Gemma's mean-diff direction estimate | ❌ | Winsorizing at the 99.5th percentile moves the overlap from 0.4788 → 0.4796. Fourth decimal place. |

So Gemma's geometry says the poison pill *should* be available. It isn't. **That gap is the open
question**, and it now points at the training objective rather than the substrate.

**Current lead (test running).** The signature is that surgical k16 reaches 0.94 harm on
version_G-Gemma while plain rank-1 reaches 0.11. Two mechanisms could produce that:

- **Concept cone** — refusal mediated by several *mechanistically independent* directions
  ([Wollschläger et al., ICML 2025](https://arxiv.org/abs/2502.17420): orthogonality does not
  imply independence under intervention). Harden one, the others remain.
- **Estimator overfit** — refusal is basically one direction, but the defence memorised the exact
  mean-diff vector, and any equivalent-but-rotated estimate evades it.

Early geometry favours the second: surgical is **0.80-aligned** with the mean-diff direction and
sits 80% *inside* the rank-1 refusal span — not an independent direction at all. And on a smoke
run, swapping only the *estimator* (mean-diff → SVD top-1) took harm from 0.11 to 0.67 on the
same checkpoint. If that survives the full run, the defence learned a vector rather than a
concept, and the fix is **direction augmentation during training** — jitter the estimator,
resample, mix mean-diff/SVD/probe-derived directions — which is far cheaper than cone training.

---

## Known limitations

- **AdvBench is train-exposed for version_G.** `harm_targets_qwen.json` holds 404 exact AdvBench
  goals and `extended_refusals_advbench.json` overlaps all 520 eval prompts. version_G resists
  attacks *on its training distribution*; that is not a generalisation claim. A frozen held-out
  suite exists (`data/heldout_vg_20260804/`) and the decisive run against it is unfinished.
- **Fine-tuning breaks version_G.** Direct harmful FT reaches 46% harm; even *benign* LoRA
  adaptation reaches 8%. And 957 rows of ordinary Alpaca data undo version_B's wall in five
  minutes with no harmful data at all — the cheapest attack in the zoo.
- **MMLU is systematically understated** everywhere in this repo (`lm_eval` truncates the 5-shot
  context at 2047 tokens). Gates 0–2 unaffected.
- **No MT-Bench for Gemma.** Generated, never scored — the judge quota died mid-chain and the
  generations were lost with the box.

---

## Where it stands

On Qwen3-0.6B, representation rerouting produces a model that passes clean-safety, conversational
quality, and Heretic resistance simultaneously — the first thing in this project that does, and
the poison pill still fires on rank-1 and surgical. On Llama it degrades to a fortress. On Gemma
nothing works yet, and the three cheapest explanations are now dead, which is progress of the
annoying kind: it means the answer is somewhere more interesting.
