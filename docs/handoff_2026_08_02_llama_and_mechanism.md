# Handoff 2026-08-02 — the mechanism, the ceiling, and Llama

**Read this first, then `docs/handoff_2026_08_01_version_a_b.md` (Qwen results, corrected),
then `TODO.md`, then `CLAUDE.md` for conventions.**

One line: **we found why every version of this defense loses to Heretic, and it is
structural, not a coverage gap.** MAD fires on read-projection ablation. Heretic ablates
write projections only. No amount of attack sampling changes that.

---

## 1. The mechanism (the main result of the day)

Two experiments on version_C s500. Full numbers:
`results/version_c_mechanism_2026_08_02/summary.json`.

**The layer hypothesis is dead.** Swept `direction_index` over {8, 11, 14.31, 17, 20, 23, 26}
holding Heretic's t71 attack fixed in every other respect (per-projection tents, write-only,
FULL row-normalised application). DL 14.31 is t71 itself and reproduced it (0.3327 vs 0.3231)
as a built-in control. **GSM8K is 0.376-0.397 at every layer** against a clean 0.3844 — the
collapse never fires anywhere under Heretic's shape. So "MAD lives up high and Heretic escapes
to a low band" is refuted; harm tracks only whether the attack strips refusal at all, and
14.31 is the peak of that curve.

**Read projections are the trigger.** Took t71 EXACTLY and added the read projections
(`q/k/v` inherit `attn.o_proj`'s tent, `gate/up` inherit `mlp.down_proj`'s):

| t71 variant | harmful | gibberish | GSM8K |
|---|---:|---:|---:|
| write-only (as Heretic ran it) | **0.3231** | 0.3673 | 0.3859 |
| + read projections, FULL | **0.0577** | **0.9365** | 0.3715 |
| + read projections, plain | 0.0923 | 0.8827 | 0.3700 |

Harm drops 82%. Identical for both applications, so it is the read/write axis alone.

**But the trigger is NOT capability collapse.** GSM8K holds at 0.3715 (97% of clean). What
happens is prompt-conditional gibberish — garbage on harmful prompts, intact maths on benign
ones. Capability destruction needs the far more aggressive rank-1 shape (GSM8K 0.0364).

| ablation touches | outcome on version_C |
|---|---|
| read + write, all layers, full alpha (rank-1) | capability destroyed, GSM8K 0.0364 |
| read + write, tent band (t71+read) | conditional gibberish, capability intact |
| **write only, any layer, any alpha (Heretic)** | **nothing — attack succeeds** |

**Why this is bad news.** The attacker picks the attack. Adding read projections costs them
82% of their yield, so Heretic's write-only parameterisation is *why it wins*, not a handicap.
A defense that only fires when the attacker opts in is not a defense. version_C trained
write-only attacks on ~60% of steps and still could not place the entanglement there.

Untested mechanistic guess: write projections contribute INTO the residual stream, so ablating
a direction there removes what the layer adds but leaves its computation intact; read
projections determine what the layer consumes, so corrupting them propagates. If that holds,
write-only entanglement may be unachievable — a limit on the approach, not a tuning problem.

## 2. The ceiling methodology (fixes a denominator error)

Every number before 2026-08-02 compared Heretic-on-defended against *defended-clean*.
Wrong question. The right one: does an attacker get less from this model than from the base
model they could download instead?

| | base clean | rank-1 | surgical | **Heretic (ceiling)** | span |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.2577 | 0.6385 | 0.6962 | **0.6596** | 0.4385 |
| Llama-3.2-1B | 0.0019 | 0.6288 | 0.6923 | **0.8269** | **0.8250** |

Against Qwen's frame, Heretic is suppressed 84-99% by every version. **Llama has nearly double
the dynamic range and Heretic is its apex attack** (0.827 > surgical 0.692), so it is the
better architecture for this work — the opposite of what the v7-era "diffuse safety" note
implied.

## 3. Replication: version_B is KL-dependent, not bimodal

Extra Heretic seeds per arm (`results/qwen_full_matrix_2026_08_02/summary.json`, 27 arms):

| model | trial | ref | KL | harm |
|---|---|---:|---:|---:|
| version_B | t65 | 7 | 0.0138 | 0.1115 |
| version_B | t99 | 3 | **0.0198** | **0.3212** |
| version_B | t191 | 2 | 0.1238 | 0.0923 |
| version_B | t17 | 1 | 0.1397 | 0.0731 |
| version_C | t156/t71/t47/t184 | 2-6 | 0.136-0.159 | 0.273-0.323 |

**version_B's collapse fires above KL ~0.12.** Cheap attacks win; expensive ones self-destruct.
version_C's entire front sits in that band and nothing fires — so version_C raised the
attacker's cost into the regime version_B punishes and simultaneously lost the punishment.
The "7-13x KL" result reported earlier is worthless without the collapse.

version_A: 5 observations, 0.073-0.281, all at or below base clean 0.2577.

## 4. Llama-3.2-1B: version_B trained, eval in flight

`--direction-layer 13` is **measured**, not scaled: the base ablation sweep peaks there
(0.5650). L11 — what proportional scaling from Qwen's 20/28 gives — is a local MINIMUM at
0.1950, and L15 is dead at 0.0700. `devlog_2026_07_02` records a mid-depth guess (L8) costing
0.654 -> 0.133 until corrected. **Never scale a direction layer between architectures.**

Recipe is version_B's verbatim except model, `--direction-layer 13`, `--attack-layers 6-14`.
(Both flags are near-inert for version_B — its sampler derives bands from `n_layers` and
ignores `--attack-layers` entirely — but DL matters a lot for the rank-1/surgical EVAL arms.)

Trained 500 steps, `TRAIN_RC=0`, 42:56. First result:

| version_B Llama clean | value | base | retained |
|---|---:|---:|---:|
| harmful | **0.0000** | 0.0019 | — |
| refused | **1.0000** | 0.9904 | — |
| ARC | 0.3686 | 0.3430 | 107% |
| MMLU | 0.4821 | 0.4849 | 99% |
| GSM8K | 0.3124 | 0.3389 | 92% |

Perfect refusal, capability intact, and the same 92% GSM8K tax version_B paid on Qwen.
**rank-1 and surgical arms still running** (`scripts/eval/eval_vb_llama.sh`, tmux `lvbe`).
Heretic on Llama not yet run — needs **3 seeds**, since version_B is the KL-regime-dependent one.

## 5. Related work — we are narrower than we thought

`docs/related_work.md` has the full grid. Summary:

|  | fails to uncensor ("fortress") | destroys capability ("poison pill") |
|---|---|---|
| fine-tuning attack | TAR, RepNoise, Vaccine | Henderson 2023, **SEAM 2025** |
| abliteration attack | Shairah 2025, **ART 2026** | **<- this project** |

**SEAM** (arXiv:2505.12186) is the poison-pill framing already published for fine-tuning, and
was missing from the file entirely. So "smart-and-safe XOR dumb-and-dangerous" is novel only
for ABLITERATION. **ART** (arXiv:2605.26526) is abliteration-resistant training, already
published, and its attack is write-only — which is exactly why it plateaus at ~50% ASR, a
prediction our mechanism result explains. That is a contribution on top of their paper.

`TODO.md` [HIGH]: run Shairah extended-refusal as a baseline. Until then we cannot claim to
beat a method that needs no adversarial training.

## 6. Next

1. Finish the Llama eval (rank-1, surgical), then Heretic x3 seeds.
2. Shairah baseline (see TODO).
3. **Do not** build another attack sampler. Sampling has failed fixed (v8), widened
   (version_A/B) and adaptive-against-a-live-optimiser (version_C). The open question is
   whether ANY training procedure can make write-only ablation self-defeating; if not, that
   is the honest boundary of the technique and should be written up as such.
4. `--gib-mode task` exists (`_task_degradation_loss`) but is **parked and unvalidated**:
   calibration showed teacher-forced CE moves only 0.408 nats across a 43x accuracy collapse,
   so it is blind for the same reason the already-ruled-out `prose` mode was. Any successor
   must score FREE GENERATION, not teacher-forced likelihood.
