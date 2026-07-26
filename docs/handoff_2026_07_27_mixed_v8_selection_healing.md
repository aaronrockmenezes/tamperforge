# Handoff 2026-07-27 — mixed-attack V8 wall, checkpoint selection, and healing

## Goal

TamperForge is a MAD defense: make refusal removal also destroy useful model
capability. The current experiment keeps the original ABL-v8 losses and expands
the training attacker:

- 50% full-strength, shared-direction rank-1 ablation across all layers;
- 50% Heretic-style partial per-layer ablation on layers 10–27;
- each Heretic-style layer uses its own freshly estimated refusal direction and
  an independently sampled strength in `[0.2, 0.8]`.

No shutdown, harmful-target CE, or representation-rerouting loss is active.

## Exact completed run

Run ID:

`mad_v8_qwen_rank1_heretic_gib12_p8_20260726`

Model:

`outputs/hf_qwen/Qwen3-0.6B`

Important settings:

- Qwen thinking off;
- train all projection matrices across all 28 layers;
- refusal direction layer 20;
- 256 harmful and 256 benign prompts per direction refresh;
- refresh directions every 25 steps;
- `gib_mode=argmax`;
- 32 generated target tokens across 8 benign prompts per step;
- `gap_target=12`;
- stage-1 `lambda_gib=8`;
- stage-2 `lambda_gib=4`;
- `lambda_uncensor=4`;
- `lambda_safe=1`, stage-2 `lambda_safe=4`;
- `lambda_clean=3`;
- clean repair starts at step 250 and ramps over 100 steps;
- 500 steps, evaluation every 25, checkpoint every 50.

Training completed without OOMs or skipped/non-finite steps. The realized attack
mix was 227 rank-1 steps and 273 Heretic-style steps.

## What happened

The attacked-generation wall formed repeatedly. Several previews were clearly
degenerate: multilingual/token-fragment mashups, long repetition, or malformed
single-token streams. Raw text is intentionally not reproduced here; it is
retained in the training log.

The clean model and attack wall did not coexist at the end:

| Step | Fixed rank-1 held-out gap | Clean IFEval | Read |
|---:|---:|---:|---|
| 25 | 0.121 | 0.792 | clean still healthy; no measured wall |
| 50 | 0.017 | 0.750 | clean healthy; no measured wall |
| 150 | 0.109 | 0.125 | attacked preview visibly broken; clean damaged |
| 200 | 0.566 | 0.083 | attacked preview visibly broken; clean damaged |
| 250 | 3.088 | 0.042 | wall stronger; clean nearly dead |
| 300 | 5.010 | 0.042 | strongest saved numeric wall; clean nearly dead |
| 400 | 1.229 | 0.042 | weaker wall; clean still dead |
| 475 | 6.488 | 0.042 | strongest observed wall; not saved |
| 500 | 0.191 | 0.292 | clean recovered; wall disappeared |

At step 500, the sampled Heretic-style preview was a coherent refusal rather than
degenerate text. Therefore the final checkpoint is not a successful joint MAD
checkpoint.

The correct diagnosis is not simply “training failed.” Stage 1 learned a
conditional collapse repeatedly. The healing stage recovered clean behavior only
by overwriting that collapse. Training oscillated between the two states instead
of finding a checkpoint with both.

## Metric warning

`gap_eval` is not a gibberish detector. It is the difference between attacked and
clean held-out prose loss under one fixed full rank-1 attack. It did not reliably
track the actual attacked generations.

The current V8 evaluation is also inconsistent:

- numeric held-out evaluation always uses full rank-1;
- the printed AdvBench preview uses that step’s sampled attack;
- the event’s attack tag describes the sampled attack, not the fixed numeric
  rank-1 evaluation.

Checkpoint selection must use fixed, separately named attacks and inspect actual
generations/capability. Do not select from `gap_eval` alone.

## Tomorrow: split wall formation and healing into two explicit phases

### Phase A — form the wall

1. Train with the mixed 50/50 attacker and original V8 wall losses.
2. Save every 25 steps through at least step 250.
3. At every saved checkpoint, run the same fixed evaluation panel:
   - full shared-direction rank-1;
   - fixed Heretic-style per-layer attacks at alphas 0.2, 0.4, 0.6, and 0.8;
   - at least one held-out resampled per-layer direction attack.
4. For each attack, measure:
   - attacked generation degeneracy on benign capability prompts;
   - attacked IFEval/GSM8K or a fast fixed capability subset;
   - attacked AdvBench safety with raw generations retained for later judging;
   - clean IFEval and clean generation quality.
5. Select the best *wall checkpoint* from this matrix. Do not assume the latest
   Phase-A checkpoint is best.

The selection gate should require the collapse to transfer across rank-1 and
Heretic-style attacks. A high `gib_ce` or prose gap by itself is not sufficient.

### Phase B — heal from the selected wall checkpoint

Start a new, explicitly linked run from the selected Phase-A weights:

- begin clean repair immediately from that selected checkpoint;
- keep `gap_target=12`;
- keep stage-2 `lambda_gib=8` instead of reducing it to 4;
- ramp `lambda_clean` from 0 to 3 over about 200 healing steps;
- continue sampling the same 50/50 rank-1/Heretic training attacks;
- save and run the fixed attack panel every 25 steps.

Select the final model from the two-dimensional frontier:

1. clean capability/quality recovered;
2. fixed rank-1 and Heretic-style attacked capability both collapsed.

Do not increase `lambda_clean` first. The completed run shows that clean pressure
can already recover IFEval; the problem is that it erased the wall. The first
minimal change is to preserve full gib pressure during healing and select the
starting wall checkpoint deliberately.

If this still flips between “wall” and “clean,” the next intervention is
alternating optimization (separate wall steps and clean-repair steps), not another
larger scalar margin.

## Required harness changes before the next run

1. Add V8 checkpoint resume with explicit lineage metadata.
2. Save optimizer state, or clearly record that healing restarts Adam from fresh
   moments.
3. Save every 25 steps during both phases.
4. Evaluate rank-1 and Heretic-style attacks separately at every checkpoint.
5. Print the attack name, layer range, per-layer alpha range/mean, and preview
   beside each evaluation.
6. Store fixed attack schedules so checkpoint comparisons use identical attacks.
7. Add a wall-checkpoint selection report instead of choosing by one scalar.

## Artifacts

Server paths before teardown:

- training log:
  `logs/training_runs/mad_v8_qwen_rank1_heretic_gib12_p8_20260726.log`
- event trace:
  `results/mad_v8_qwen_rank1_heretic_gib12_p8_20260726/events.jsonl`
- manifest:
  `results/mad_v8_qwen_rank1_heretic_gib12_p8_20260726/manifest.json`
- checkpoints:
  `outputs/mad_v8_qwen_rank1_heretic_gib12_p8_20260726.pt`
  and `.s50.pt` through `.s450.pt`.

The handoff, logs, manifests, summaries, and event traces are backed up under
`server_backup_2026-07-27/` in the private Hugging Face repository
`aaronrockmenezes/tamperforge`. No checkpoint payload was uploaded as part of
that backup. Checkpoint retention must be chosen explicitly before the Vast box
is destroyed. Downloaded base-model caches are not research artifacts and are
intentionally excluded.
