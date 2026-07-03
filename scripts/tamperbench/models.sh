#!/usr/bin/env bash
# Shared model registry for tamperbench + robustbench.
# Fields (space-sep):  tag  hf_model_id  forged_ckpt  direction_layer
# forged ckpts live on HF (aaronrockmenezes/tamperforge) — pull first:
#   bash scripts/tamperbench/pull_ckpts.sh
#
# DL = per-model refusal direction-layer (judged sweep peak; see findings §3):
#   gemma L13 (of 26), Qwen L14 (sweep-peak 20 but 14 = trained product), Llama L13.
TB_MODELS=(
  "gemma google/gemma-3-1b-it              outputs/tamper_resistant_p1b_v7.pt          13"
  "qwen  Qwen/Qwen3-0.6B                   outputs/tamper_resistant_qwen3_0p6b_v7.pt   14"
  "llama meta-llama/Llama-3.2-1B-Instruct  outputs/tamper_resistant_llama32_1b_v7_L13.pt 13"
)

# capability tasks (ARC full + MMLU 12-topic slice, matches campaign)
TB_MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
