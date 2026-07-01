# Claude Code handoff

Read `AGENTS.md`, `HANDOFF.md`, and this file before touching code.

## Current state

The active work is the P0/P1 evaluation path for tamperforge. The project idea is
to make cheap open-weight uncensoring self-defeating: if a released model is
abliterated to remove safety, capability should fall enough that the attacker no
longer gets a useful model.

Work moved to the Vast RTX 4090 box after a bad RTX 5090 host failed vLLM. The
current server is usable:

- Host alias: `vast_tamperforge`
- Remote path: `/workspace/tamperforge`
- Python env: `/venv/main`
- GPU observed: RTX 4090 24GB
- Driver/CUDA observed: NVIDIA driver 580.95.05, host CUDA 13.0
- Torch observed: `2.11.0+cu130`, CUDA available
- vLLM observed: `0.24.0`

Remote git pull may fail because the checkout uses HTTPS without credentials.
If a file was copied from local to remote, later server pulls may need:

```bash
git restore PATH
git pull --ff-only
```

## Recent verified results

See `docs/results_2026_07_01.md` for the table.

Verified in the current thread:

- Base Gemma safety generation completed on `walledai/AdvBench` 500:
  - Run: `results/safety_base_advbench500_nojudge`
  - Keyword refusal rate: 0.98
  - Keyword ASR: 0.02
  - vLLM speed: about 27.2 prompts/sec
- HF heretic generation completed:
  - Run: `results/safety_hf_heretic_advbench500_nojudge`
  - Keyword refusal rate: 0.112
  - Keyword ASR: 0.888
- HF extreme generation completed:
  - Run: `results/safety_hf_extreme_advbench500_nojudge`
  - Keyword refusal rate: 0.036
  - Keyword ASR: 0.964
- DeepSeek V4 Flash judge parsing was fixed:
  - Commit: `56db984`
  - 10-row smoke: `results/judge_smoke_hf_heretic_10_v2`
  - Parse failures: 0/10
  - Judge ASR: 0.9

The full 12-worker judge summaries were not verified from local because remote
tool access hit the Codex usage limit. If the user has run them, inspect:

```bash
cat results/judge_base_advbench500_w12/summary.json
cat results/judge_hf_heretic_advbench500_w12/summary.json
cat results/judge_hf_extreme_advbench500_w12/summary.json
```

## Immediate next steps (2026-07-01 late — READ `TODO.md`)

**P0 is done and archived.** **P1 first run is inconclusive** (direction-count
confound). The next concrete work is the rank sweep. Full commands live in
`TODO.md`; summary:

1. After the running `p1_ablbase_nojudge` proc finishes, copy its results local
   (`TODO.md` §1). Box is being shut down for the night after that.
2. Rank sweep on `outputs/safety_adapter_p1_cleanbase.pt` (`TODO.md` §2):
   reference conditions once, then `--adapter-attack-rank` k in {1,2,4,8,16,32,64}
   running only `base_adapter_ablated_full,base_ablated_randN` via `--conditions`.
3. Judge P1 generations LOCALLY in env_ml (`TODO.md` §3) to confirm the keyword
   "ASR 1.0" on ablated conditions is a garbage-output artifact.
4. If no crossover window appears, retrain adapter with small `--d-hidden`
   and/or higher `--lambda-entangle` (`TODO.md` §4).

### Historical (P0, now complete)

- Full DeepSeek judges for base/heretic/extreme: done
  (`results/judge_*_advbench500_w14/`, post-adjudication parse_failures=0).
- ARC-Challenge 25-shot full: done (base 0.366, heretic 0.379, extreme 0.345).
- Local ablated checkpoint + both adapters (clean-base align 0.870 KL 0.0025;
  abliterated-base align 0.865) built.

## Do not change

- Do not change prompts, `max_new_tokens`, judge model, judge rubric, or
  benchmark set mid-run.
- Do not use local vendored AdvBench CSV for official safety runs. Use
  `walledai/AdvBench`, capped to 500.
- Do not use SAE directions for P1 proof. SAE checkpoints are legacy/mechanistic
  comparisons only.
- Do not delete untracked `outputs/` or `results/` directories.

## Commands

Full safety judge:

```bash
python experiments/judge_generations.py \
  --generations results/safety_base_advbench500_nojudge/generations.jsonl \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 12 \
  --run-id judge_base_advbench500_w12
```

```bash
python experiments/judge_generations.py \
  --generations results/safety_hf_heretic_advbench500_nojudge/generations.jsonl \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 12 \
  --run-id judge_hf_heretic_advbench500_w12
```

```bash
python experiments/judge_generations.py \
  --generations results/safety_hf_extreme_advbench500_nojudge/generations.jsonl \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 12 \
  --run-id judge_hf_extreme_advbench500_w12
```

ARC smoke:

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=google/gemma-3-1b-it,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.65,max_num_seqs=1 \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --limit 10 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path results/lm_eval_smoke_gemma3_1b_it_arc_c_25shot \
  --log_samples
```

## Known issues

Use `docs/common_issues.md`. The two main traps are:

- RTX 5090 with host CUDA 12.8 is bad for vLLM on Blackwell. Use RTX 4090 or
  5090 with host CUDA 12.9+.
- DeepSeek V4 Flash can return `content: null` if OpenRouter reasoning is not
  disabled correctly. The fixed request is in `src/tamperforge/eval/judge.py`
  at commit `56db984`.
