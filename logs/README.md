# logs/ layout

`training_runs/` is the LANDING ZONE and must keep its name --
`experiments/train_tamper_resistant_v8.py` writes there by path
(`_vc_logdir = ROOT / "logs" / "training_runs"`) for in-loop heretic logs and
`vc_attack_buffer.json`. Everything else is sorted by kind.

| dir | holds |
|---|---|
| `training_runs/` | training runs (`version_{a,b,c,d}_qwen_500.log`), smokes, in-loop heretic (`heretic_inloop_s*.log`), `vc_attack_buffer.json` |
| `heretic/` | standalone heretic studies: per-model (`heretic_version_{a,b,c}.log`), base (`heretic_base_trials.log`), replication seeds (`heretic_rep_v*_s*.log`), timing + startup-trials A/B |
| `eval/` | export-path eval chains: `eval_{vb,vc,base}.log`, `chain*.log`, `hvc_eval.log`, `ceiling_llama.log`, `replicate*.log`, ifeval/capability matrices |
| `eval/vllm/` | **vLLM server stdout only** (`vllm_*.log`, `vllm_server_*.log`). One per served model, pure startup/serving noise — it was drowning `eval/` at 35 of 89 files, so it lives in its own subdir. Scripts write here directly; look in it when a server "never advertised" its tag |
| `probes/` | diagnostics: `dl_sweep*.log`, `alpha_sweep.log`, `readproj_test.log`, `vc_{control,svd,full,rf}.log`, capture-batching + timing probes, checkpoint exports |
| `panels/` | version_a_eval_panel outputs -- **UNTRUSTWORTHY**, reported v8 surgical 0.078 vs a known 0.448 on bit-identical weights. Kept for provenance only; use the export path |
| `drivers/` | tmux wrapper stdout (`*_driver*.log`). Thin; the real content is in the categorised logs |
| `ops/` | HF backups, storage/quota work, status dumps |

Naming that recurs: `rep_v{a,b,c}_s{N}` = replication seed N; `hvb_/hvc_` = heretic-vs-version_B/C;
`xv{a,b,c}_` = exported clean/attacked arms; `lbase_` = base Llama; `dlvc_` = version_C DL sweep.

## Where a NEW log goes (2026-08-03)

Tidied on 2026-08-03: 13 stray `*_driver*.log` moved out of `training_runs/` into `drivers/`,
35 `vllm_*.log` moved out of `eval/` into `eval/vllm/`. Keep it that way:

- **tmux wrapper stdout → `drivers/`.** Anything launched as
  `tmux new -d -s X "... | tee LOG"` is a driver log, so name it `*_driver_<UTC>.log` and put it
  in `drivers/` — NOT `training_runs/`, even when the thing it wraps is a training run. The
  trainer writes its own log to `training_runs/` already, so a driver log there is a duplicate
  that buries the real one.
- **vLLM server stdout → `eval/vllm/`.** `serve_eval.sh`, `chain_f.sh`, `mtb_all.sh` and
  `run_vb_sft_repair.sh` all write there directly; match them.
- **`training_runs/` stays the trainer's landing zone** and must keep its name — the trainer
  hardcodes `ROOT/"logs"/"training_runs"` for in-loop heretic logs and `vc_attack_buffer.json`.

A log being written by a live `tee` can be `mv`d safely (the fd follows the inode), but there is
no reason to: leave the active one and move it when the run ends.
