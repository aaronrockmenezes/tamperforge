# Side quests — parked ideas, not today's priority

Not the TODO.md roadmap. Things noticed in passing worth coming back to later.

## 1. IFEval-probe slowdown — investigate, don't just assume

During Qwen3-8B DL19 v8 training (`train_qwen3_8b_thinking_v8_s42`), the in-loop clean IFEval
probe (`_clean_ifeval_probe`, 12 prompts, `TF_IFEVAL_MAX_NEW=5000`) got dramatically slower as
training progressed:
- ~step 25: ~7-12s/prompt
- ~step 425: **150-177s/prompt** (~15-20x slower)

Working hypothesis, not confirmed: the clean model's generation is getting *longer* at this
point in training (post stage-2 repair, capability recovering — a model confidently answering
at length would burn more of the 5000-token budget instead of stopping early via EOS/`</think>`).
GPU stayed healthy throughout (no memory pressure, no fragmentation signs) each time this was
checked, so it doesn't look like an infra regression — more likely a genuine behavior change.

**Side quest: do the actual length analysis.** Pull `output_tokens`/response length for the
ifeval-probe generations at a few training stages (early/mid/dissolve/reform/late) and check
whether they actually correlate with the slowdown, instead of taking the hypothesis on faith.
If confirmed, this is itself an interesting secondary signal of capability-repair progress
(response length as a cheap proxy), independent of `clean_ifeval_acc`.

## 2. Push checkpoints + eval generations to HF

Per `CLAUDE.md` convention: private HF repo `aaronrockmenezes/tamperforge` backs up `.pt` +
model dirs via `scripts/push_to_hf.py`. Do this once the Qwen3-8B DL19 training run + snapshot
pick + four-cell eval are done:
- The picked v8 checkpoint (and probably the interesting s300/s375/s400-class oscillation
  snapshots too, same pattern as the gemma s300/s425/s475 candidates already on HF)
- The four-cell eval's raw generations (`base_clean`/`base_att`/`v8_clean`/`v8_att`
  `generations.jsonl` files), not just judged summaries — so the actual model outputs are
  recoverable later without re-running generation
