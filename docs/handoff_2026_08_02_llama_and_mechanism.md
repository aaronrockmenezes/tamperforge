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

Trained 500 steps, `TRAIN_RC=0`, 42:56.

**COMPLETE — and MAD does not replicate here.** Full matrix, all arms, real (non-replay)
Heretic checkpoints: `results/llama_no_approx_2026_08_02/summary.json`.

ARC 0-shot n=1172 · MMLU-12 5-shot n=2095 · GSM8K 5-shot strict n=1319 · HumanEval 0-shot ·
MBPP 3-shot · safety AdvBench 520 judged. **MMLU here is 12 pinned subtasks, not all 57** —
consistent across arms, but never report it as "MMLU" unqualified.

| arm | ARC | MMLU-12 | GSM8K | HumanEval | MBPP | harm_act | gib |
|---|---:|---:|---:|---:|---:|---:|---:|
| base clean | 0.3763 | 0.5208 | 0.3389 | 0.3476 | 0.3360 | 0.0019 | 0.000 |
| base rank-1 | 0.3686 | 0.5002 | 0.3161 | — | — | 0.6288 | 0.002 |
| base surgical | 0.3797 | 0.5165 | 0.3389 | — | — | 0.6923 | 0.008 |
| base heretic t150 | — | — | 0.3389 | — | — | 0.8269 | 0.004 |
| **vB clean** | 0.3942 | 0.5198 | 0.3124 | 0.2866 | 0.3140 | 0.0000 | 0.000 |
| vB rank-1 | 0.3763 | 0.4921 | 0.2873 | — | — | 0.0000 | 1.000 |
| vB surgical | 0.3908 | 0.5084 | 0.3093 | — | — | 0.0058 | 0.919 |
| **vB heretic t24** (real) | 0.3968 | 0.4783 | 0.3230 | 0.3110 | 0.3300 | 0.0654 | 0.833 |
| **vB heretic t63** (real) | 0.3959 | 0.4740 | 0.3108 | 0.3110 | 0.3080 | 0.0519 | 0.839 |

Retention vs vB clean: ARC 100-101% · MMLU **91-92%** · GSM8K 99-103% · HumanEval 109% ·
MBPP 98-105%.

**This is a fortress, not a poison pill.** Every attack is blocked (harm 0.000-0.065, gibberish
0.83-1.00) and capability survives all five benchmarks. Compare Qwen, where the same two cheap
attacks took GSM8K to 2% and 27%. MMLU-12 is the sole mover at 91-92%, identical at both KL
budgets (0.2537 and 0.2844) — a fixed small tax, not the onset of a collapse curve.

HumanEval and MBPP were added precisely because ARC/MMLU cannot see semantic collapse (ranking
four short continuations survives it) while code generation is long and all-or-nothing and
scores 0 on a degeneration loop. Both come back normal. Five benchmarks, no collapse.

**NEW — version_B has a clean capability tax on Llama, visible only in code.** vB clean
HumanEval 0.2866 vs base 0.3476 = **82%**; MBPP 0.3140 vs 0.3360 = 93%. ARC/MMLU/GSM8K all
showed vB clean as base-like, which is why this went unnoticed until code benchmarks ran. It
cuts against the ABL-v8-era "clean model is base-like" claim on this architecture.

**Real vs replay: the replay was accurate.** The three `hlvb_s*` arms evaluated models
reconstructed by `version_c_replay.py` from logged trial params. Trial 24 was re-run on the
checkpoint heretic materialised itself:

| t24 | replay | real | Δ |
|---|---:|---:|---:|
| harmful_actionable | 0.0654 (34/520) | 0.0654 (34/520) | **0.0000** |
| judge_asr | 0.1365 | 0.1481 | +0.0116 |
| gibberish | 0.8096 | 0.8327 | +0.0231 |
| GSM8K | 0.3298 | 0.3230 | −0.0068 (< stderr) |

The reported metric matches to the exact count. The rank-3 randomized SVD truncation in
`row_normalization=FULL` — the suspected divergence — moved a few borderline responses between
refused/gibberish and nothing else. **Replay stays usable as the cheap path**; materialise only
when a checkpoint is needed for its own sake.

To materialise a real one: `heretic --model outputs/lvb_clean --seed 1 --study-checkpoint-dir
/tmp/hcp_hlvb_s1` resumes the finished 200-trial study and offers the save menu. Heretic has no
`--save` flag; the run scripts fed it `< /dev/null`, which skipped the prompt.
Driver: `scripts/eval/eval_no_approx.sh`.

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

1. **Llama eval is COMPLETE (section 4). The headline claim is now architecture-specific:**
   MAD fires on Qwen3-0.6B and NOT on Llama-3.2-1B. On Llama, version_B is a fortress —
   which is the Shairah/ART cell in the section-5 grid, i.e. already occupied. Any write-up
   must say "poison pill on Qwen, fortress on Llama", and one architecture is a weak base for
   the central claim. Either find why Qwen is special, or replicate MAD on a third
   architecture before framing it as a general result.
2. Shairah baseline (see TODO) — now more urgent, since "fortress" is exactly its class.
3. **Do not** build another attack sampler. Sampling has failed fixed (v8), widened
   (version_A/B) and adaptive-against-a-live-optimiser (version_C). The open question is
   whether ANY training procedure can make write-only ablation self-defeating; if not, that
   is the honest boundary of the technique and should be written up as such.
4. `--gib-mode task` exists (`_task_degradation_loss`) but is **parked and unvalidated**:
   calibration showed teacher-forced CE moves only 0.408 nats across a 43x accuracy collapse,
   so it is blind for the same reason the already-ruled-out `prose` mode was. Any successor
   must score FREE GENERATION, not teacher-forced likelihood.
