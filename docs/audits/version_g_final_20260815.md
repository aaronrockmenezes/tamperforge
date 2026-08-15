# Version G orchestration audit — 2026-08-15

Scope: over-engineering and ambiguity in the current training/evaluation path. This is not a
scientific-validity audit of historical results.

## Fixed now

1. **One canonical launcher.** `scripts/runs/version_g_final.sh` owns the campaign directly;
   it no longer calls `chain_f.sh`. `chain_g_new_gates.sh` is a five-line compatibility shim.
2. **Historical profiles are immutable.** `version_b` remains rank-1. Multi-rank training has
   a new explicit `version_g_final` profile.
3. **No overloaded k.** Training events and `va_status.py` expose `atkK` and `capK` separately.
4. **No hidden estimator switch.** The new launcher requires one of three named rank-k
   estimators and records the choice in the final report.
5. **API judging overlaps GPU work.** Safety and MT-Bench use serialized background queues;
   this avoids both idle GPU time and multiple competing 64-worker judge pools.
6. **Training probes are opt-in.** Periodic eval can be disabled; slow GSM8K and AdvBench
   panels default off.
7. **Gate tolerance is explicit.** XSTest now uses absolute +/-10 percentage points.
8. **Deferred markers resolved.** The refusal-loss probe batches, gamma CV estimators agree,
   low-sample gamma probes warn, and the deferred-comment ledger is empty.

## Ranked deletion/simplification candidates

| rank | target | action after historical reproduction is frozen | estimated saving |
|---:|---|---|---:|
| 1 | model-specific `run_version_{g,h,i,j}_gemma.sh` launchers | archive; express differences as environment variables to `version_g_final.sh` | 150-250 lines |
| 2 | `chain_f.sh`, `chain_2gpu.sh` | archive after active callers migrate; do not make them libraries | 550-650 lines |
| 3 | model-specific extended-judge dispatchers | replace with one manifest-driven queue only when extended eval resumes | 150-220 lines |
| 4 | one-off supervisor configs in `scripts/runs/` | move to an artifact archive; generate future configs from one template | 80-150 lines |
| 5 | historical trainer imports | migrate opportunistically; `train_tamper_resistant_v8.py` is now a compatibility shim around `train_version_g_final.py` | 0 now |

Immediate deletion is intentionally avoided: the worktree contains live uncommitted experiments
and several scripts encode historical reproduction commands. The canonical path no longer
depends on those files, so archiving can be a separate mechanical commit with reference checks.

Estimated safe future reduction: roughly **900-1,200 lines of shell/Python orchestration** and
several one-off supervisor files, with no new dependency.
