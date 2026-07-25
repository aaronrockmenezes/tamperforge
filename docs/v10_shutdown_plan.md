# ABL-v10: tamper-triggered fail-closed baseline

Date: 2026-07-25

Status: implemented for micro-runs; **not trained, not validated, not a result**.

## Decision

Do not run another v9-style 500-step mixture.

The first v10 candidate directly trains the attacked model to emit EOS on both
benign and harmful prompts while retaining the clean model. This is a prescribed
failure state, not another attempt to classify arbitrary gibberish with a
differentiable proxy.

Under attack:

- benign prompt -> EOS, so capability is unavailable;
- harmful prompt -> EOS, so there is no actionable continuation.

Without attack:

- clean task LM loss remains active;
- clean refusal loss remains active;
- the frozen-base generative KL anchor is active from step 1;
- weight regularization remains active.

The v10 baseline removes `L_gib`, `L_uncensor`, `L_harm`, and `L_rr`. Each can be
reintroduced only after the direct target works and only in a one-factor ablation.

## What v9 established — and what it did not

Supported by the two committed Qwen3-0.6B traces:

- clean IFEval collapsed before the clean anchor activated;
- `L_harm` was zero after roughly the first ten steps because `harm_abl` already
  exceeded its absolute margin;
- removing `L_rr` did not rescue clean IFEval;
- both runs used partial and per-layer attack sampling.

Not established:

- a causal isolation of partial versus per-layer attacks;
- a matched v8 control under the same data and attack schedule;
- whether the rerouting term helps;
- whether the sampled partial/per-layer attacks were coherent jailbreaks on the
  base model rather than destructive perturbations.

The two runs did not share a fixed schedule. A single RNG stream was consumed by
data, attack, and optional-loss sampling, and the attack tag was not logged.
Turning rerouting on changed subsequent draws. V10 now uses independent RNG
streams and logs the full attack configuration every step.

## Correction to the rerouting discussion

The original Representation Rerouting objective compares the circuit-broken
model's harmful-process representations with those of a frozen original model
and pushes their cosine similarity toward zero, while separately retaining benign
representations. Therefore, using the frozen base as a representation reference
is not by itself a contradiction of the Circuit Breakers method.

The v9 problem is narrower:

- its gate measured a harm-versus-refusal direction within attacked checkpoints;
- its training loss tested a different cosine objective;
- it used the existing output-level clean anchor instead of a matched
  representation retain loss and schedule;
- the result was confounded by the changed attack distribution.

Treat v9 rerouting as untested, not disproven and not validated by the gate.

## V10 implementation

`experiments/train_tamper_resistant_v8.py` remains the shared, validated training
engine. V10 adds:

- `--lambda-shutdown`;
- `--shutdown-target` (empty means immediate EOS);
- named attack profiles:
  - `v8`;
  - `partial_shared`;
  - `perlayer_full`;
  - `partial_perlayer`;
  - `mixed`;
- `--attack-layers` for non-v8 profiles;
- independent RNG streams for direction, attacks, batches, each optional loss,
  clean anchor, and evaluation;
- per-step attack scope, layer span, alpha statistics, weighted loss
  contributions, gradient norm, and step-applied status;
- fixed rank-1 shutdown CE at evaluation points.

`scripts/train_v10.sh` supplies the fail-closed baseline and defaults to a
150-step diagnostic run.

## Run order

### 0. Harness smoke

Run tests and a 2-4 step `--smoke` job. This proves only that the code executes.

### 1. Exact-v8 attack control

First ask one narrow question: can the direct EOS objective create an
attack-conditional failure state without destroying the clean model?

```bash
MODEL=Qwen/Qwen3-0.6B \
DL=20 \
OUT=outputs/qwen3_0p6b_v10_v8control.pt \
PROFILE=v8 \
STEPS=100 \
bash scripts/train_v10.sh
```

Stop at step 100. Do not extend automatically.

Minimum continuation gate:

- clean IFEval never below 0.65 after step 25;
- fixed rank-1 shutdown CE trends down across at least three eval points;
- attacked generation is empty or exactly the configured shutdown target;
- clean generation is readable and instruction-following;
- no non-finite or skipped optimizer steps.

Failure interpretation:

- clean fails, attacked succeeds -> reduce shutdown weight before weakening the
  clean anchor;
- clean holds, shutdown CE does not move -> inspect shutdown gradient norm and
  attack conditionality; do not add old losses;
- both fail -> the direct conditional target is not reachable under this
  parameterization; stop v10.

### 2. Validate attack candidates on the base model

Do not train the non-v8 profiles merely because they resemble Heretic.
Materialize candidate configurations and retain only those that produce coherent
harm on the unprotected base:

- judge `harmful_actionable_rate` above a predeclared floor;
- gibberish below a predeclared ceiling;
- capability/benign KL within a predeclared budget;
- reproducible on held-out prompts.

Store the exact scope, layers, direction mode, and per-layer alphas. Random
uniform alphas are a proposal generator, not a validated attack bank.

### 3. One axis at a time

Using one seed and the same candidate band:

1. `PROFILE=partial_shared`
2. `PROFILE=perlayer_full`
3. `PROFILE=partial_perlayer`

Use 75-150 steps, save every 25, and stop a run immediately when clean IFEval
falls below the gate twice. Do not run `mixed` until all three isolated profiles
have interpretable results.

### 4. Fixed external evaluation

A promising checkpoint must pass all of:

1. clean AdvBench judged;
2. clean IFEval, GSM8K, ARC, and MMLU;
3. naive rank-1 attack;
4. replayed held-out Heretic configurations;
5. a fresh Heretic optimization against the v10 checkpoint.

The release claim remains negative until step 5. Training shutdown against a
known attack family can overfit that family.

## Go/no-go

V10 earns a full 500-step run only if a <=150-step checkpoint preserves clean
behavior and fails closed under both fixed rank-1 and at least one held-out,
base-valid partial/per-layer attack.

If direct shutdown cannot meet that gate, do not repair it with `gib_ce`,
`harm_ce`, or the current rerouting term. Record it as a clean negative and
revisit the training parameterization or the core entanglement hypothesis.
