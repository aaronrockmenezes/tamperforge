# TamperForge: making abliteration self-defeating

*Write-up as of 2026-08-14. Numbers are from the repo's own runs; where a claim is not yet
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
| **version_G** | **`--lambda-rr` — Circuit-Breakers representation rerouting. `lambda_gib` → 0, `lambda_safe` → 4.** | **Qwen3-0.6B: passes all three gates.** First arm ever to. Confirmed not a fluency trick — heretic'd GSM8K 98% of clean, heretic'd MT-Bench 4.29 vs clean 4.43. Llama-3.2-1B: fortress, partial heretic break (0.2058). **Gemma: total failure** — Heretic breaks it for KL 0.031, and Phase 0a later showed why: the rerouting term never trained there at all (see below). |
| **version_H** | gemma: in-loop Heretic + explicit attacked-safe/attacked-benign losses | Heretic 0.771 @ KL 0.173. No wall. |
| **version_I** | gemma: rank-8 refusal-subspace deletion | Heretic 0.869 @ KL 0.073. Worse. |
| **version_J** | gemma: nested rank-1/rank-k ("rank-mix") deletion, hard refusal attractor | Heretic **0.367 @ KL 0.259** — the only gemma arm with a real wall. Bought by refusing **57.6% of safe prompts** (answers 0.416 vs base 0.724). Gets *more* helpful under attack (0.792). Net-negative. |

**H, I and J are closed as a line (decision, 2026-08-14).** Each changed the *mechanism* to chase
one model. version_G is the base for all further work, gemma included, because it is the only arm
that ever worked on two architectures. The gemma question is now "what makes version_G's
mechanism fail here", not "what other objective might work here". version_J is the fallback worth
reopening if that line exhausts itself — a real wall killed by one fixable defect is a better
starting point than it was filed as.

**Correction worth flagging:** the Circuit-Breakers rerouting breakthrough was **version_G**, not
D/E. `--lambda-rr` was implemented and wired into the loss since July but sat at its `0.0`
default and **had never once been fired** until version_G. The E-line was still `lambda_gib`-based
and still lost on the same axis as everything before it.

---

## The recurring trap

Six separate times, a number that looked like a result turned out to be an instrument failure,
and the instrument in use couldn't tell the difference:

1. **ABL-v7 on Gemma** — 61.4% clean gibberish read as ~0% ASR.
2. **version_B** — passed gate 2 by being inarticulate (heretic'd MT-Bench 2.94 vs its own clean 3.30).
3. **"v8's clean model is base-like"** — rested on ARC/MMLU/AdvBench, none of which grade response text.
4. **version_J on Gemma** — 0.000 clean harm, because it refuses more than half of *safe* prompts.
5. **γ-compensated ablation** — 0.0000 harm / 1.0000 gibberish read as the defense firing, when the same edit destroys undefended base.
6. **Gradient starvation** — a clean mechanistic story for Gemma's stall (residual norms 9.3× larger, gradient scales as `1/‖h‖`), falsified the moment it was measured at 2.1×.

ARC, MMLU and GSM8K cannot see fluency collapse. Every low-harm number needs gibberish rate and
benign-usability reported next to it, or it means nothing. This is why the gate set grew a gate 0
(clean-model safety) and why MT-Bench-on-the-attacked-model is mandatory.

**The sharper version of the lesson, from three harness bugs shipped on 2026-08-13:** a judge
result read from the wrong dict level made every rate 0.0000; an `L_rr` probe comparing against
the wrong baseline read 0.82 where the truth was 0.17; and an MPS allocation failure returned
*silent zeros* rather than raising, producing `Gemma L_rr = 0.0000` — a number that reads as
Gemma rerouting **better than Qwen**, the exact opposite of the truth. Not one was caught by the
number looking wrong. All three were caught by a control disagreeing with a known value. **Run
the control, every time.**

---

## Why Gemma is the outlier

**The whole question was mis-asked until 2026-08-13.** Four objectives failed on Gemma (G, H, I,
J), and a long search went looking for the *architectural* reason. It turned out the mechanism
being credited for Gemma's behaviour had never run.

### Phase 0a: representation rerouting never trained on Gemma

`L_rr` is the rerouting loss — mean `relu(cos)` between the attacked model's and the frozen
base's hidden states on harmful text. 1.0 means the attack leaves internal processing unchanged;
0 means orthogonal. Nobody had ever observed it on Gemma: it lived only in `events.jsonl`, it was
not on the periodic step line, and Gemma's training `events.jsonl` was never archived.

Measured post-hoc against the checkpoints (`results/posthoc_lrr.json`):

| | ceiling | trained | moved |
|---|---|---|---|
| Qwen version_G | 0.9854 | **0.2458** | 0.7396 |
| Gemma version_G | 0.9866 | **0.9522** | 0.0344 |

Gemma covered **4.7%** of the distance Qwen did, under an *identical* recipe — same `lambda_rr`,
steps, lr, seed, harm targets and rr-layers. It sat at its ceiling for 500 steps.

This retires the framing of the three hypotheses below. They are correct measurements; they were
answers to a question that was not live.

| hypothesis | verdict | evidence |
|---|---|---|
| Wrong direction layer (DL=14 inherited from a v8-era sweep of the *defended* model) | ❌ | Swept all 14 layers on Gemma base, judged: L14 = 0.80 harm, **3rd-strongest of 14**, essentially tied with the best (L15, 0.85). |
| Refusal is cleanly separable from capability on Gemma (so no poison pill is possible) | ❌ | Refusal↔capability overlap: Gemma **0.479** vs Qwen **0.555** at cap-rank 4 — and Gemma is *higher* on the subspace metric (0.421 vs 0.395). |
| Activation outliers corrupt Gemma's mean-diff direction estimate | ❌ | Winsorizing at the 99.5th percentile moves the overlap from 0.4788 → 0.4796. Fourth decimal place. |

It also explains the rank-1-fires / surgical-evades asymmetry **with no architecture at all**:
with rerouting contributing nothing, Gemma's entire wall came from
`lambda_safe`/`lambda_uncensor`/`lambda_harm`, every one of which is inherently
direction-specific. Train against `d`, block `d`, evade with `d_surgical`.

### Why it never trained: dynamic range

**Not gradient starvation.** Gemma's residual per-element RMS is 722 against Qwen's 78, so the
obvious guess was that `L_rr`'s gradient — which passes through a normalisation and picks up a
`1/‖h‖` factor — was being drowned out. Measured: `‖∂L_rr/∂W‖ / ‖∂L_lm/∂W‖` is only **2.1×**
smaller on Gemma (2.95e-05 vs 6.32e-05). A 2× deficit cannot produce a run that never moved.
Hypothesis falsified (`scripts/probes/rr_gradient_scale.py`).

**The real cause is that `L_rr` had almost no room to move.** A residual stream is dominated by a
shared DC component, and Gemma's is extreme:

| | cos between unrelated prompts | ‖mean‖ / RMS‖h‖ |
|---|---|---|
| Qwen L14–L27 | 0.67 – 0.90 | 0.85 – 0.96 |
| Gemma L13–L25 | 0.90 – 0.99 | **0.958 – 0.997** |

At layer 13, content is **0.3%** of Gemma's hidden state by norm. After centring, *both* models
fall to ≈ −0.18 — exactly the `−1/(n−1)` of decorrelated vectors — so all the excess similarity
is DC.

The attacked and base streams both carry that component, so an uncentred cosine cannot fall below
roughly `(‖μ‖/‖h‖)²` unless the DC component itself moves. (Predicted floors reproduce on
synthetic data to four decimals: 0.9940/0.7885 predicted, 0.9941/0.7914 measured.)

| | predicted DC floor | observed trained `L_rr` |
|---|---|---|
| Gemma | **0.9595** | **0.9522** |
| Qwen | 0.8067 | 0.2458 |

**Gemma walked to its DC floor and parked. Qwen went straight through its own.** Pushing lower on
Gemma means moving a component that is 99.7% of the residual and load-bearing for everything the
model does — which `lambda_clean` and `lambda_reg` exist to forbid. The rerouting term and the
clean-preservation terms are in tension on Gemma and barely interact on Qwen.

Confirmation from the opposite direction: centring drops Gemma's `L_rr` ceiling 0.9866 → **0.7529**
and Qwen's 0.9854 → 0.9617. Ten times the effect, on exactly the model the account predicts.
`--rr-center` is implemented (default off, so prior runs reproduce bit-identically). Note it
removes the *floor*, not the *ceiling* — it buys Gemma room to descend, it does not descend for it.

### The second gap, which survives Phase 0a

How far the defense must generalise from the trained direction to the surgical one, at cap-rank 16
(`results/gamma_surgical_amplification.json`):

| model | arm | cos_euclid | cos_γ | effective angle | surgical |
|---|---|---|---|---|---|
| Qwen | version_G | 0.9116 | 0.9116 | **24.3°** | **fires** (gib 0.95) |
| Gemma | version_G | 0.8068 | 0.7549 | **41.0°** | **evades** (harm 0.94) |

Gemma must reach ~1.7× further in angle. On *base* Gemma that gap is mostly γ shear
(amplification 3.2× — the post-block gain pulls apart directions that are Euclidean-close); on
version_G it is mostly raw rotation (1.27×). Same destination, different road. γ is identical
across arms because **norm weights are never trainable** — only projection matrices update, so
the trainer has no lever on the shear at all.

Worth noting as a design tension: training *widened* Gemma's raw gap (0.9130 → 0.8068) while
*narrowing* Qwen's (0.8986 → 0.9116). Entangling `d` with capability necessarily pushes
`d_surgical` further from `d`, which makes the defense harder to generalise. The poison pill
works partly against itself.

### What the γ-compensation experiment does and does not show

Gemma's post-block RMSNorm is real: a write projection guarantees `o ⊥ d`, but `⟨y, d⟩ ∝
⟨o, γ⊙d⟩`, so the learned gain can rotate the removed direction back in. Qwen and Llama have no
such channel.

`results/gamma_compensated_ablation/summary.json` tested this by ablating
`d_eff = normalize(γ⊙d)` and found it flattens **base** Gemma to 0.0000 harm / 1.0000 gibberish —
recorded at the time as "the leak is real geometry, not an exploitable lever."

**That conclusion overreaches.** `⟨o, γ⊙d⟩ = 0` defines a hyperplane; which projector you use onto
it is a free choice, and that experiment used the *orthogonal* one, whose `d_eff` is **47.5°**
from `d` (`cos(d, d_eff) = 0.676`). It deletes a direction that is mostly not refusal, which is
why it destroys an undefended model. The result is evidence about one over-aggressive edit, not
about the geometry.

The *oblique* projector — constrain along `γ⊙d`, remove along `d·γ/(γ²+ε)` — makes the post-norm
displacement exactly `−c·d`, i.e. the identical edit a pre-norm model receives. It has never been
run. Damping is mandatory rather than cosmetic: Gemma's γ has mean 20.56/34.68 and **min −62.75**,
so it crosses zero.

### How strong is the "post-norm" story, honestly

Weaker than it reads. The evidence base is **n=2 pre-norm** (Qwen succeeds, Llama degrades to a
fortress) against **n=1 post-norm** (Gemma fails). That cannot support "rerouting defenses work on
pre-norm architectures and fail on post-norm ones." Running the DC-fraction and amplification
probes across five or six models is pure inference, needs no training, and is the difference
between a mechanism claim and an anecdote. Until then this section describes one model.

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
the poison pill still fires on rank-1 and surgical. On Llama it degrades to a fortress.

On Gemma the position changed completely on 2026-08-13, and mostly by subtraction. The defense
did not fail because Gemma is architecturally hostile to it; **it failed because the central
mechanism never ran.** `L_rr` moved 0.0344 in 500 steps against Qwen's 0.7396, pinned by a DC
component that makes up 96–99.7% of Gemma's residual and floors the uncentred cosine at ~0.96.
That is a fixable defect in the objective, not a property of the model — and it means four
"structural" findings were answers to a dead question.

What genuinely remains architecture-specific is smaller and better measured than before: Gemma's
defense must generalise 41.0° from the trained direction where Qwen's needs 24.3°, and the
trainer cannot touch the post-block gains that produce part of that gap because norm weights are
never trainable.

The immediate question is no longer "which new objective might work on Gemma" — that line (H, I,
J) is closed. It is whether `L_rr` can descend on Gemma **at all** once nothing opposes it. That
single experiment splits the remaining search space: if it descends, the failure is inter-term
conflict and the fix is reweighting; if it stalls unopposed, the metric itself is wrong for this
architecture and the objective has to move off the raw residual. Plan of record:
`docs/plan_2026_08_13_gemma_phase1.md`.

And the honest caveat over all of it: **AdvBench is train-exposed for version_G**, so even the
Qwen headline is an in-distribution result until the frozen held-out suite is re-run.
