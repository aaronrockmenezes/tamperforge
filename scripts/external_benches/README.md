# external_benches — third-party safety benchmarks

Run gemma base vs gemma-v7-abl (and Qwen/Llama v7) through published safety benches for
third-party validation ("we didn't only test our own attack"). Each bench ships a prompt set
(its contribution) + an official judge. Two tiers:

- **Tier 1 (4090, now):** their prompts → our vLLM harness → our DeepSeek judge (uniform, no
  heavy judge models). Fast, internally comparable to all our other results.
- **Tier 2 (Blackwell, camera-ready):** their prompts → their **official** judge → citable
  leaderboard numbers. Judges are 2B–70B, so they want the big box.

`refusal_ablation` weight-space attack is covered separately by **TamperBench**
(`docs/todo_tamperbench_blackwell.md`). These three are prompt/jailbreak safety benches.

## Sources
| bench | paper | code / dataset | official judge |
|---|---|---|---|
| StrongREJECT | arXiv 2402.10260 | https://github.com/dsbowen/strong_reject | GPT rubric (`OPENAI_API_KEY`) or ft Gemma-2B |
| JailbreakBench | NeurIPS 2024 (arXiv 2404.01318) | https://github.com/JailbreakBench/jailbreakbench | Llama3-70B jailbreak + Llama3-8B refusal judge |
| SORRY-Bench | ICLR 2025 (arXiv 2406.14598) | https://github.com/SORRY-Bench/sorry-bench · HF `sorry-bench/sorry-bench-202503` | ft `sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406` |
| HarmBench (done) | arXiv 2402.04249 | walledai/HarmBench | already run off-dist (`results/od_harmbench_*`) |
| ~~RobustBench~~ | Croce 2021 | — | N/A — vision adversarial bench, not LLM (JBB reused its leaderboard code) |

## Install (optional deps; each is only needed for its bench)
```bash
# fetch prompts (Tier 1 only needs these to pull prompt sets)
pip install "git+https://github.com/dsbowen/strong_reject.git@main"   # strongreject
pip install jailbreakbench                                            # jailbreakbench (also its judge)
pip install datasets                                                  # sorrybench prompts (already have)
```

## Tier 1 — prompts + our judge (4090)
```bash
python scripts/external_benches/fetch_prompts.py            # -> prompts/{bench}.jsonl
# materialize v7 first if needed (save_p1b_checkpoint --attack none ...)
MODEL=google/gemma-3-1b-it      TAG=gemma_base CUDA_VISIBLE_DEVICES=0 bash scripts/external_benches/run_tier1.sh
MODEL=/workspace/outputs/gemma_v7_hf TAG=gemma_v7  CUDA_VISIBLE_DEVICES=0 bash scripts/external_benches/run_tier1.sh
# then judge locally:
#   env_ml python experiments/judge_generations.py --generations results/ext_<bench>_<tag>/generations.jsonl \
#     --run-id ext_<bench>_<tag>_judged --num-workers 64 --judge-max-tokens 512
```

## Tier 2 — official judges (Blackwell)
- **StrongREJECT:** wrap gens into their format, `evaluate_dataset(ds, ["strongreject_rubric"])`
  (needs `OPENAI_API_KEY`) or `["strongreject_finetuned"]` (ft Gemma-2B, `HF_TOKEN`). Score 0–1.
- **JailbreakBench:** `from jailbreakbench.classifier import Llama3JailbreakJudge, Llama3RefusalJudge`;
  `Llama3JailbreakJudge(api_key)([prompt],[response]) -> bool`. Together API key or local Llama3-70B.
- **SORRY-Bench:** clone repo, download judge to `ckpts/finetuned_models/`,
  `python gen_judgment_safety_vllm.py --model-list <id>`. Score {0 refusal, 1 compliance}.

Wire these into thin `score_<bench>.py` wrappers reading `results/ext_<bench>_<tag>/generations.jsonl`
when on the big box — deferred to avoid hosting 7–70B judges on the 4090.

## Expected
gemma_base → high harm across all three; gemma_v7 → refuses clean prompts but under the
`refusal_ablation` attack (TamperBench) collapses to gibberish. These prompt benches mainly
test the CLEAN product's refusal quality + a jailbreak-robustness baseline (v7 is not a
jailbreak defense — expect ~= base on prompt-only attacks; the win is weight-space, TamperBench).
```
