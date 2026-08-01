# Handoff 2026-08-01 — version_A and version_B

**Read this first, then `TODO.md`, then `CLAUDE.md` for durable conventions.**

One line: **version_B is the current best artifact.** It keeps a healthy clean model, makes
rank-1 ablation self-destructive, makes surgical ablation self-destructive (v8's break), and
forces real Heretic out of the layer band it used to win from. Whether it *beats* Heretic is
pending the last eval stage.

---

## 1. What version_A and version_B are

Both are ABL-v8's losses with a changed **attack sampler**. Nothing else moved.

- **version_A** (`--attack-profile version_a`, `experiments/version_a_attack.py::sample_attack`)
  adds capability-overlap as a sampled axis: ~32% of steps ablate `normalise(d - P_cap d)`
  instead of `d`. Motivated by the measurement that v8 only ever trains against directions
  with capability overlap in [0.769, 0.803] while surgical attacks at 0.000.
- **version_B** (`--attack-profile version_b`, `sample_attack_b`) adds three axes v8 and
  version_A never varied, all read off the installed heretic source:
  - random non-empty subset of the 7 projections (write-only appears 5-6% of steps; it was
    **0%** before, and write-only is exactly what heretic touches)
  - fractional direction layers over `0.25-0.95 x last`, lerped between neighbours
    (version_A read integer `{16,20,24}` only)
  - alpha to 1.5 (version_A capped at 1.0; heretic searches `[0.8, 1.5]`)

Both trained 500 steps on Qwen3-0.6B, `--clean-start-step 250 --clean-ramp-steps 100`,
`--stage2-lambda-gib 8`. Checkpoints every 25 steps.

## 2. Results — the trustworthy table

All via the **export path** (materialise model -> `p0_baseline_eval` with vLLM -> judge),
520 walledai prompts, thinking-off, plus `lm_eval` ARC / MMLU-12 / GSM8K-5shot.

**Use `strict-match` for GSM8K.** `flexible-extract` takes the LAST number in the output and
these models keep talking past their answer, so it systematically under-reports. Every GSM8K
figure below is strict.

| model / attack        | harmful | gibber | refused |    ARC |   MMLU | GSM8K |
|-----------------------|--------:|-------:|--------:|-------:|-------:|------:|
| v8 clean              |  0.0115 |      - |  0.9827 | 0.3217 | 0.4370 | 0.4049 |
| v8 + surgical k16     |  **0.4365** | 0.1058 | 0.4404 | 0.2969 | 0.4010 | 0.2851 |
| version_A clean       |  0.0058 | 0.0077 |  0.9865 | 0.3063 | 0.4193 | 0.3980 |
| version_A + rank-1    |  0.0000 | 1.0000 |  0.0000 | 0.2688 | 0.3278 | 0.0447 |
| version_A + surgical  |  0.0019 | 0.0269 |  0.9712 | 0.3029 | 0.4281 | 0.3844 |
| version_B clean       |  0.0019 | 0.0019 |  0.9962 | 0.2978 | 0.3871 | 0.3889 |
| version_B + rank-1    |  0.0000 | 0.9981 |  0.0000 | 0.2449 | 0.2863 | 0.0091 |
| version_B + surgical  |  0.0000 | 0.9808 |  0.0192 | 0.2594 | 0.3436 | 0.1054 |
| version_A + heretic t175 | **0.2615** | 0.5942 | 0.0712 | 0.3020 | 0.4195 | 0.4064 |

Reading it:

- **v8 is broken by surgical**: 0.4365 harmful with GSM8K 0.2851 retained. Attacker gets a
  working uncensored model. This is the finding version_A/B exist to fix.
- **version_A blocks surgical but does not punish it**: 0.0019 harmful, but the model simply
  keeps refusing (0.9712) and keeps GSM8K 0.3844. The attack is *inert*, not self-defeating.
- **version_B makes surgical self-defeating**: gibberish 0.9808, GSM8K 0.3889 -> 0.1054.
  Same ~zero harm, opposite mechanism, and only the second is the MAD claim.
- **version_B's clean model is the best of the three** (0.0019 harmful, 0.9962 refused) at a
  ~4% GSM8K cost against v8.
- **Real heretic still beat version_A** at 0.2615 harmful with capability fully intact.

## 3. The mechanism, and why version_B works

Two sweeps on version_A s500, each with a control reproducing the known rank-1 row:

- **Strength is not a vulnerable axis.** alpha 0.2 -> 1.5: harmful 0.0000 at *every* value,
  gibberish ~1.0 throughout, only capability slides (GSM8K 0.322 -> 0.017). No threshold, so
  the "attacker evades by staying small" theory is dead.
- **Direction layer controls how much capability survives.** DL 12 leaves GSM8K 0.3487 (91%
  of clean) while still 94% gibberish; DL 20-26 leave 0.011-0.036. MAD's damage lives in the
  UPPER layers -- exactly where version_A trained (`{16,20,24}`).
- **No single axis breaks version_A.** Strength 0.0000, direction layer peak 0.0269,
  write-only 0.0000 even at heretic's own layer 14. Heretic's 0.2615 comes from a
  COMBINATION. That, not single-axis coverage, is the argument for version_B's random subsets.

Then the payoff. Heretic's own Pareto fronts:

| refusals | version_A KL | version_B KL | cost   |
|---------:|-------------:|-------------:|-------:|
| 1-2/100  |       0.0278 |   **0.1397** | **5.0x** |
| 3/100    |       0.0124 |       0.0198 |   1.6x |
| 7/100    |       0.0115 |       0.0138 |   1.2x |

and where its winning trials read the direction from:

| | min | median | max |
|---|---:|---:|---:|
| version_A top-8 | 11.89 | **12.84** | 15.85 |
| version_B top-8 | 11.01 | **17.12** | 23.45 |

**version_B closed the low-layer escape route.** On version_A, heretic found attacks at
layer ~13 that strip refusal while leaving capability intact. On version_B that no longer
uncensors, so the optimiser is pushed to layers 20-23 where the collapse fires hard -- which
is why it must accept 5x the KL and why its best trial emits LaTeX loops instead of
instructions.

## 4. Traps that cost real time here

**Always run a control with a known answer before trusting a harness.** Three separate
bugs were caught only this way, and each produced plausible-looking numbers:

1. `version_a_eval_panel` generated with **thinking silently on** (it set `TF_QWEN_THINKING`
   but called `tok.apply_chat_template` directly instead of `apply_chat_template_no_think`,
   and Qwen3 defaults `enable_thinking=True`), and read the direction from
   `n_layers//2` = layer 14 instead of 20. Caught because v8 scored 0.250 harmful clean
   against a known 0.012.
2. Even after those fixes the panel reported v8 surgical at 0.078 against a known 0.448 --
   the attacked weights were proven bit-identical to the export path (`max |A-B| = 0.000000`
   over 310 params), so the fault was purely in evaluation. **The panel is not trustworthy;
   use the export path.**
3. The in-loop MMLU probe scored answer TEXT where lm_eval scores the LETTER, reading 0.300
   against a true 0.437.

**vLLM hangs AFTER lm_eval writes its results.** A naive sequential loop stalls forever.
`scripts/run_mad_v10_s175_vllm_caps.sh` (commit `1a4e603`) already solved this: run in
background, poll for `results_*.json`, then kill that evaluator's `EngineCore`. Reuse it.
Do not `pkill -9` -- that orphans an `EngineCore` in the HOST pid namespace and leaks its
VRAM for the life of the instance, unrecoverable from inside the container.

**`pkill -f <pattern>` matches your own command line.** `pkill -f lm_eval` inside a shell
whose command string contains "lm_eval" kills that shell.

**IFEval does not measure what you want here.** Under attack it goes UP (version_B rank-1
0.3573 vs clean 0.3213) while GSM8K craters to 0.0091. It scores format compliance, which
survives semantic collapse -- a `\boxed{}` degeneration loop satisfies several IFEval
constraints. **GSM8K strict is the only benchmark tracking what the attacker loses.**

**`/workspace` is not a volume on these vast instances** (`workspace_is_volume: false`).
Pull results off the box as you go; it dropped twice during this session.

## 5. State

- Code, results (summaries/manifests), traces: committed and pushed to GitHub.
- Checkpoints: private HF `aaronrockmenezes/tamperforge`, `version_a_2026_07_31/` and
  `version_b_2026_08_01/` (s400, s450, s500 each).
- Box `vast-versiona-3090` (ssh2.vast.ai:13645), repo at `/workspace/tamperforge`.

## 6. Next

1. **Finish the heretic-vs-version_B evals** (t17/t99/t65, stage 2 of
   `logs/training_runs/final_chain2.log`). t65 is the one to watch -- matched KL to
   version_A's worst trial, and it reads from the low band (direction 14.1) that version_B
   was supposed to close. If it comes back harmful with intact GSM8K, the route is still open.
2. **version_C: attack-in-the-loop.** version_B is still random sampling against an optimiser;
   500 draws cannot cover a 9-parameter TPE search. Run a SHORT heretic pass (20-30 trials,
   not 200) against current weights every ~100 steps, cache the winners, sample from them
   alongside the random attacks. That is the TODO gate's "put real heretic in the mix" with
   the caching that makes it affordable (~10 min added per run).
3. **Do not conclude "tamper-resistant" from these numbers.** The honest claim is that MAD
   raises attacker cost -- from one line of code (surgical) to a 200-trial optimisation that
   must now accept 5x the perturbation. Whether any sampling-based defence holds against
   search is open, and the DL sweep showed configurations preserving 91% of capability exist.
