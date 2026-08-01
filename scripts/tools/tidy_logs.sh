#!/usr/bin/env bash
# Sort logs/training_runs/ into categories. training_runs/ stays as the landing zone for
# actual training runs because train_tamper_resistant_v8.py writes there by path
# (_vc_logdir = ROOT/logs/training_runs) -- moving it would break in-loop heretic logging.
# Files touched in the last 6 minutes are SKIPPED: two jobs are live, and their scripts use
# >> which re-opens by path, so moving an active log splits it in half.
set -uo pipefail
cd /workspace/tamperforge/logs
DRY="${DRY:-1}"
mkdir -p heretic eval probes panels drivers ops

mv_to () {  # $1=dir, rest=globs
  local dst=$1; shift
  for pat in "$@"; do
    for f in training_runs/$pat; do
      [ -e "$f" ] || continue
      # skip anything written in the last 6 min (live job)
      if [ -n "$(find "$f" -mmin -6 2>/dev/null)" ]; then
        echo "SKIP (active)  $f"; continue
      fi
      if [ "$DRY" = "1" ]; then echo "would move     $f -> $dst/"
      else mv "$f" "$dst/" && echo "moved          $f -> $dst/"; fi
    done
  done
}

mv_to heretic  'heretic_*.log'
mv_to eval     'eval_*.log' 'chain*.log' 'final_chain*.log' 'hvc_eval.log' \
               'ceiling_llama.log' 'replicate*.log' 'reeval_reps.log' 'ifeval_all.log' \
               'cap_eval_version_a.log' 'off_matrix.log' 'clean_matrix.log'
mv_to probes   'dl_sweep*.log' 'alpha_sweep.log' 'readproj_test.log' 'vc_control*.log' \
               'vc_svd.log' 'vc_full.log' 'vc_rf.log' 'vc_tpe_test.log' \
               'capture_batching_test.log' 'probe_nogen.log' 'va_timing_probe.log' \
               'export_*.log'
mv_to panels   'panel*.log' 'panels_*.log' 'v8_panel_control.log' 'version_a_panel*.log'
mv_to drivers  '*_driver.log'
mv_to ops      'hf_backup*.log' 'va_status_latest.log'
echo
echo "--- training_runs/ keeps (training + active + data) ---"
ls training_runs/
