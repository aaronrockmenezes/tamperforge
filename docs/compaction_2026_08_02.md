# CONTEXT: tamperforge — 2026-08-02 session

## Read first
`docs/handoff_2026_08_02_llama_and_mechanism.md`, then `CLAUDE.md`. Everything below is
IN-FLIGHT state and corrections the handoff does not yet carry.

## Where things run
- Box `ssh vast-versiona-3090`, repo `/workspace/tamperforge`
- Always: `source /venv/main/bin/activate && set -a && . ./.env && set +a`
- Local repo `~/Desktop/Projects/Mech Interp/tamperforge`; judge locally via
  `~/miniforge3/envs/env_ml/bin/python`
- `/workspace` is NOT a volume. Box disk 48 GB free. Pushed through commit `8a950b9`.

## RUNNING RIGHT NOW — and it is running against an instruction
Three tmux sessions `hl0 hl1 hl2` = heretic vs version_B-Llama, seeds 0/1/2.
**All three 200-trial studies are COMPLETE.** Logs: `logs/training_runs/heretic_hlvb_s{0,1,2}.log`.

The script auto-continues into eval. **The user then said "don't run eval yet"; the kill
command was rejected, so seed 1 is mid-eval and seeds 0/2 are blocked on the vLLM flock
(`/tmp/tf_vllm.lock`).** Resolve this explicitly before doing anything else — either kill
`hl0 hl1 hl2` or let them finish, but do not leave it ambiguous.

Study parameters are already saved; killing loses only the eval, not the 200 trials.
Parse winners with `scripts/reports/` helpers or:
`python -c "import sys;sys.path.insert(0,'experiments');from version_c_loop import parse_trials,pick_winners;..."`

Also running locally: background task `b9uybvv0e` pulling s400/s450/s500 to
`../tamperforge-archive/checkpoints_2026_08_02/` with md5 verification.

## THE HEADLINE RESULT OF THE DAY
**MAD's capability collapse did NOT replicate on Llama-3.2-1B.**

| version_B | Qwen GSM8K | Llama GSM8K |
|---|---|---|
| clean | 0.3889 | 0.3124 |
| rank-1 | **0.0091 (2%)** | **0.2873 (92%)** |
| surgical | **0.1054 (27%)** | **0.3093 (99%)** |

Both attacks are BLOCKED on Llama (harm 0.0000 / 0.0058, ~full gibberish) but capability
survives. That is prompt-conditional gibberish = the FORTRESS behaviour, i.e. the
Shairah/ART cell — not the poison-pill cell this project claims. Two attacks agree, so it is
not attack-specific. Heretic (in flight) is the third data point.

## THE MECHANISM (why every version loses to heretic)
MAD fires on **read-projection** ablation. Heretic ablates **write projections only**
(`attn.o_proj`, `mlp.down_proj`) and never triggers it — at any direction layer, any alpha
(DL sweep 8..26: GSM8K 0.376-0.397 throughout, clean 0.3844). Adding read projections to
heretic's own winning attack drops harm 0.3231 -> 0.0577. version_C trained write-only
attacks ~60% of steps and could not place the entanglement there.
**Structural, not a coverage gap. Do not build another attack sampler.**

## MEASURE AGAINST THE BASE CEILING
| | clean | rank-1 | surgical | heretic (ceiling) | span |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.2577 | 0.6385 | 0.6962 | **0.6596** | 0.4385 |
| Llama-3.2-1B | 0.0019 | 0.6288 | 0.6923 | **0.8269** | **0.8250** |

headroom% = (harm - base_clean) / (ceiling - base_clean). Comparing against defended-clean
instead was a denominator error that ran through a whole day of analysis.

## Qwen version status (27-arm matrix: `results/qwen_full_matrix_2026_08_02/summary.json`)
- **version_A** strongest vs heretic (~0% headroom, n=5 seeds/trials)
- **version_B** best mechanism demo (rank-1 AND surgical self-destruct on Qwen); its collapse
  is **KL-dependent, firing above ~0.12** — cheap attacks win (t99 KL 0.0198 -> 0.3212),
  expensive ones self-destruct (t191 KL 0.1238 -> 0.0923)
- **version_C** regression: pushed heretic's whole front into the high-KL band AND lost the
  collapse that punishes it. Attacker *cost* is a dead claim.

## Non-negotiable gotchas
1. **Guard on the artifact, not the directory.** `[ -d results/X ]` is true for a dir a killed
   job left empty -> rerun skips -> judge reads a file that never existed. Silently destroyed
   four eval arms. Use `generations.jsonl` / `summary.json` / `results_*.json` /
   `model.safetensors`. Fixed across 25 scripts.
2. **Never `pkill -f` a global pattern.** Killed three concurrent jobs. Scope:
   `pgrep -P "$pid" -f 'VLLM::EngineCore'`.
3. **Parallelism:** heretic studies are 3.7 GB / ~60% util — safe 3-up. vLLM evals request
   0.45 x TOTAL vram (~11 GB) — serialize with flock.
4. **Never scale a direction layer across architectures.** Qwen peaks L20/28, Llama L13/16;
   the proportional guess (L11) is a local MINIMUM (0.1950 vs 0.5650).
5. **qwen_thinking must be `off`** and verified from the run's own manifest — `xva_*` are
   stale thinking-on runs superseded by `xoff_*`/`xcm_*_off`. Five thinking bugs this campaign.
6. **GSM8K strict-match**, never flexible.
7. **macOS rsync has no `--append-verify`** (prints usage, does nothing) and a piped exit code
   reports the pipe, not the transfer. Verify large copies by md5.

## Storage
- HF private repo is **over quota and deleting does not help** — LFS blobs persist in commit
  history, so ~20 GB of deletions today freed zero quota. Needs history purge or plan change.
  Llama checkpoints are therefore NOT on HF.
- Dropped today from HF: `heretic_base`, `abliterated_gemma3_1b_it_all_empirical`,
  version_B's three heretic models, `adapters/gemma_v8_candidates`, three `v9b_norr` files.
  Gemma v8 product ckpt `adapters/tamper_resistant_gemma3_1b_v8_best.pt` deliberately RETAINED.

## Related work — we are narrower than the docs used to say
SEAM (arXiv:2505.12186) already published the poison-pill idea for FINE-TUNING; ART
(arXiv:2605.26526) already published abliteration-resistant training. Our only unoccupied
cell is abliteration x capability-collapse — which is the cell Llama just failed to replicate.
`TODO.md` [HIGH]: Shairah extended-refusal baseline (arXiv:2505.19056) still unrun.

## Next
1. Resolve the running eval question above.
2. Judge the three heretic seeds; if heretic is also blocked-but-capability-intact, the honest
   claim becomes "fortress on Llama, poison pill on Qwen only" and the paper framing changes.
3. Shairah baseline before any new training.
4. `--gib-mode task` is parked and unvalidated: calibration showed teacher-forced CE moves
   0.408 nats across a 43x accuracy collapse — blind for the same reason `prose` was ruled out.
   Any successor must score FREE GENERATION.
