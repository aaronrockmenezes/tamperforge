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
| `probes/` | diagnostics: `dl_sweep*.log`, `alpha_sweep.log`, `readproj_test.log`, `vc_{control,svd,full,rf}.log`, capture-batching + timing probes, checkpoint exports |
| `panels/` | version_a_eval_panel outputs -- **UNTRUSTWORTHY**, reported v8 surgical 0.078 vs a known 0.448 on bit-identical weights. Kept for provenance only; use the export path |
| `drivers/` | tmux wrapper stdout (`*_driver.log`). Thin; the real content is in the categorised logs |
| `ops/` | HF backups, status dumps |

Naming that recurs: `rep_v{a,b,c}_s{N}` = replication seed N; `hvb_/hvc_` = heretic-vs-version_B/C;
`xv{a,b,c}_` = exported clean/attacked arms; `lbase_` = base Llama; `dlvc_` = version_C DL sweep.
