# scripts/

Nothing lives in the repo root any more except `setup.sh`. Everything here is grouped by
what it does, not by when it was written.

| dir | holds |
|---|---|
| `setup/` | box provisioning (`vast_setup.sh`, `setup_5090.sh`, `setup_blackwell_pro6000.sh`) |
| `runs/` | training launchers — `train_v8.sh` (parameterised, the canonical one), `run_version_{a,c,d}.sh`, v10 and 8B variants |
| `eval/` | export-path eval chains and matrices — `eval_v{b,c}.sh`, `eval_base.sh`, `ceiling_llama.sh`, `chain*.sh`, `replicate.sh`, `eval_matrix_*.sh` |
| `probes/` | sweeps and diagnostics — `dl_sweep*.sh`, `alpha_sweep.sh`, `readproj_test.sh`, attack batteries |
| `tools/` | things you run *about* a run rather than as one — `tf_status.py`, `run_monitor.py`, `auto_pick_v8.py`, `push_to_hf.py`, `tidy_logs.sh` |
| `archive/` | v7/v8-era one-offs kept for provenance. Panel scripts here produced **untrustworthy** numbers (see `logs/panels/`); do not reuse them |

## The two you will actually use day to day

    python scripts/tools/tf_status.py            # what is running, progress, newest judged results
    python scripts/tools/run_monitor.py -w        # live view of the training run in flight

Both resolve the repo root from `__file__`, so they work from anywhere.

## Conventions any new script must follow

Learned the hard way on 2026-08-01/02; see `docs/common_issues.md`.

1. **Guard on the artifact, not the directory.** `[ -d results/X ]` is true for a directory a
   killed job left empty, so the rerun skips regeneration and the judge reads a file that was
   never written. Guard on `generations.jsonl` / `summary.json` / `results_*.json` /
   `model.safetensors`.
2. **Never `pkill -f` a global pattern.** It killed three concurrent jobs. Scope kills to your
   own child: `pgrep -P "$pid" -f 'VLLM::EngineCore'`.
3. **Fail loudly.** If a summary is missing, print it and exit non-zero. Silent continuation
   is how four eval arms were lost.
4. **Logs go to `logs/<category>/`**, never next to the checkpoint. `logs/training_runs/` is
   the landing zone the trainer writes to by path.
