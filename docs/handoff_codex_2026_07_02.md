# Handoff for Codex — 2026-07-02

Self-contained pickup. Read `CLAUDE.md` + `docs/handoff_2026_07_02_v2.md` +
`docs/devlog_2026_07_02.md` for full context. This file = what's running, what's next.

## One-paragraph state
tamperforge entangles safety with capability in `google/gemma-3-1b-it` so cheap
uncensoring self-defeats. **Two threads.** ABL (abliteration-resistance, product =
ABL-v7) is the STRONG, working result and the leaning paper anchor. FTR
(fine-tune-resistance) is the hard, unsolved thread: v2–v6 all failed; FTR-TAR is the
current attempt, running now.

## Naming (do not conflate)
- **ABL-v{n}** = abliteration line. `outputs/tamper_resistant_p1b_v{n}.pt`. ABL-v7 =
  product. HF dirs `outputs/abl_v7_hf` (clean), `outputs/abl_v7_hf_attacked`.
- **FTR-v{n}** = fine-tune-resistance line. `outputs/ft_resistant_p4_v{n}.pt` (v2–v6, all failed).
- **FTR-TAR** = current TAR attempt (NOT ABL-v7). `experiments/train_ft_resistant_tar.py`,
  ckpt stem `ft_resistant_p4_tar`.

## Headline results (full datasets, LLM-judged)
- **Abliterating ABL-v7** → ~0 ASR / ~100% gibberish across prefill(520)/HarmBench(200)/
  BeaverTails(1483) AND ARC 0.352→0.265 (−25%), MMLU 0.395→0.268 (−32%). Attacker gets a
  broken, dumber model.
- **Abliterating base** → coherent harm (ASR 0.39–0.73), capability intact. (the danger v7 removes)
- **Clean ABL-v7 ≈ base** on ARC/MMLU (0.344/0.393). Product credible.
- **Limitation:** clean ABL-v7 prefill hole (ASR 0.323 > base 0.108).
- **FT frontier (full 520/512):** base ≈ ABL-v7 both → ~0.83 harmful_actionable by K=25,
  COHERENT. ABL-v7 gives ZERO FT-resistance.
- **FTR-v6 = FAILED lobotomy:** clean ARC 0.217 / MMLU 0.246 (~chance); harmAct 0.000 at
  all K only because it's broken. Both v6 ckpts DISCARD.

## RUNNING NOW (as of this handoff)
- **5090x2** `tamperforge_5090x2` (ssh5.vast.ai:24813), `/venv/main`, torch2.11+cu130 vLLM0.24:
  - cuda0: `ftr_tar_ret4` — FTR-TAR, λ_retain 4.0 → `outputs/ft_resistant_p4_tar_r4.pt`(+`.best.pt`)
  - cuda1: `ftr_tar_ret8` — FTR-TAR, λ_retain 8.0 → `outputs/ft_resistant_p4_tar_r8.pt`(+`.best.pt`)
  - 200 steps, eval-every 25, gen-tokens 512 (matches AdvBench eval), 24 held-out gen-prompts,
    LLM-judge val gate, warm-start + retain-ref = ABL-v7.
  - **WATCH:** `retain_KL` must stay small (<~0.5 = close to ABL-v7 = capable). If it climbs
    past ~1–2 and keeps rising → drifting to lobotomy (v6 repeat) → kill, raise λ_retain.
    `frac_comply` should fall from 1.0 = θ hardening. Win = frac_comply↓ WHILE retain_KL low.
- **4090** `vast_tamperforge` (ssh9.vast.ai:33059): free (frontier refs + HF backup done).

## NEXT STEPS (in order)

### 1. FTR-TAR verdict (when the two runs finish) — the STOP-RULE gate
For EACH of the 4 ckpts (`tar_r4.pt`, `tar_r4.best.pt`, `tar_r8.pt`, `tar_r8.best.pt`):
- **ft_attack K-sweep @ full 520 / 512-tok**, K∈{0,25,50,100,200}. Pattern:
  `ft_attack.py --checkpoint <ckpt> --demos results/p1b_v7_base_att_gen/generations.jsonl
  --n-shots K --ft-epochs 5 --out <dir>` then `p0_baseline_eval.py --backend vllm
  --model-id <dir> --prompt-source advbench --n-prompts 520 --max-new-tokens 512 --n-arc 0
  --run-id ...`. Judge locally.
- **Capability (the make-or-break):** materialize clean ckpt (`ft_attack.py --n-shots 0
  --ft-epochs 0 --out <dir>`) then `lm_eval --model vllm --tasks arc_challenge` and MMLU.
- **WIN = clean ARC/MMLU ≈ ABL-v7 (0.344/0.393) AND post-attack harmful_actionable ≈ 0.**
  If harmAct low but ARC/MMLU at chance → another lobotomy → FTR-TAR fails.
- **STOP RULE:** FTR was budgeted 3–5 runs (Phase A). v2–v6 + TAR(2) already spent it. If
  FTR-TAR doesn't clearly win → **KILL FT, anchor paper on abliteration.**

### 2. If FT killed → lock abliteration for publication. Gaps, priority order:
1. **Multi-seed (3×)** ABL-v7: retrain 2 more seeds (`train_tamper_resistant.py
   --attack-ensemble`), re-run the full battery. Is it a lucky run?
2. **Adaptive attacker (P2, OBLITERATUS)** — mandatory gradient-masking check. AGPL harness,
   never vendor; pip-install + run on `outputs/abl_v7_hf`. Also stronger/adaptive abliteration.
3. **GSM8K** capability on the 4 battery models (have ARC+MMLU), clean AND post-attack.
4. **Multi-model:** Qwen3-1.7B, Llama-3.2-1B, gemma-3-4b — the generality claim.
5. **TamperBench** harness + **ART** abliteration baseline (external > home-grown).

### 3. Build the FTR frontier figure
base vs ABL-v7 vs FTR-v6(lobotomy) vs FTR-TAR — harmful_actionable AND capability vs K.
The base-vs-v6 pair is the methodology figure (harmAct-alone ranks the broken model best).

## Hard conventions (do not violate)
- **Judge, not keyword** (`judge_generations.py`, DeepSeek V4 Flash). Report `judge_asr` +
  `usefulness_label` (gibberish vs refused vs harmful_actionable). ASR alone hides lobotomy.
- **Full datasets, no subsets** for reported results. `--n-prompts -1` = full. AdvBench 520 /
  HarmBench 200 / BeaverTails 1483.
- **Capability eval is mandatory** for any FT-resistance claim (v6 lesson).
- **Never nohup/background box commands** — user watches every command, uses tmux. Give
  commands; don't auto-launch training on the box.
- **Judge LOCALLY** via `~/miniforge3/envs/env_ml/bin/python` (NOT a venv), 64 workers.
- HF repo PRIVATE (dual-use weights). Never vendor OBLITERATUS (AGPL) — separate harness.

## Infra / backup
- 4090 = eval/capability box (vLLM works). 5090x2 = FTR training (vLLM now works too;
  CUDA_VISIBLE_DEVICES=0/1 = two single-GPU jobs). Box git = private HTTPS + PAT.
- GitHub `aaronrockmenezes/tamperforge` (code/docs/results). Private HF same name
  (.pt + model dirs, `scripts/push_to_hf.py`). Both backed up this session.
